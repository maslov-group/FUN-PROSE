from copy import deepcopy
import os
import pandas as pd
import ray
from ray import tune
from ray.tune.suggest.skopt import SkOptSearch
from scipy.stats import pearsonr
import torch
from torch.utils.data import Dataset, DataLoader

from data import JsonDataset

### Device set-up
use_cuda = True

if use_cuda:
    print("using gpu")
    cuda = torch.device('cuda')
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    def cudaify(model):
        return model.cuda()
    scaler = torch.cuda.amp.GradScaler
else: 
    print("using cpu")
    cuda = torch.device('cpu')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    def cudaify(model):
        return model
    def MockScaler():
        def scale(self, x):
            return x
        def step(self, optim):
            optimizer.step()
        def update(self):
            pass
    scaler = MockScaler

class Vocab:
    """
    A simple vocabulary class that takes an alphabet of symbols,
    and assigns each symbol a unique integer id., e.g.
    
    > v = Vocab(['a','b','c','d'])
    > v('c')
    2
    > v('a')
    0
    > len(v)
    4
    
    """
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.index_map = {letter: index for (index, letter) in 
                          list(enumerate(alphabet))}
        
    def __len__(self):
        return len(self.alphabet)
        
    def __call__(self, letter):
        return self.index_map[letter]

class Tensorize:
    """
    An instance of Tensorize is a function that maps a piece of data
    (i.e. a dictionary) to an input and output tensor for consumption by
    a neural network.
    """
    
    def __init__(self, symbol_vocab, max_word_length, category_vocab):
        self.symbol_vocab = symbol_vocab
        self.category_vocab = category_vocab
        self.max_word_length = max_word_length
    
    def __call__(self, data):
        words = Tensorize.words_to_tensor(data['seq'], 
                                              self.symbol_vocab, 
                                              self.max_word_length).float()
        
        #print(len(data['tfs']))
        #print(torch.stack(data['tfs']).shape)
        tfs = torch.stack(data['tfs'],dim=1).float()
        #print(type(data['exp']))
        #label = LongTensor([self.category_vocab(c) 
        #                     for c in data['exp']])
        label = data['exp'].float()
        return cudaify(words), cudaify(tfs), cudaify(label)
        
    @staticmethod
    def words_to_tensor(words, vocab, max_word_length):
        """
        Turns an K-length list of words into a <K, len(vocab), max_word_length>
        tensor.
    
        e.g.
            t = words_to_tensor(['BAD', 'GAB'], Vocab('ABCDEFG'), 3)
            # t[0] is a matrix representations of 'BAD', where the jth
            # column is a one-hot vector for the jth letter
            print(t[0])
    
        """
        tensor = torch.zeros(len(words), len(vocab), max_word_length)
        for i, word in enumerate(words):
            for li, letter in enumerate(word[len(word) - max_word_length:]):
                tensor[i][vocab(letter)][li] = 1
        return tensor

class ProCNN(torch.nn.Module):
    """
    A two layer CNN that uses max pooling to extract information from
    the sequence input, a one layer fully-connected network to extract TF information 
    then trains a fully connected neural network on the to make predictions.
    
    """    
    def __init__(self, config, output_classes, input_symbol_vocab):
        super(ProCNN, self).__init__()
        
        self.config = config
        self.input_symbol_vocab = input_symbol_vocab
        
        self.xt_hidden = config["xt_hidden"]
        self.xs_hidden = config["seq_length"] // (config["pool1"] * config["pool2"])

        self.output_classes = output_classes
        
        self.conv1 = torch.nn.Conv1d(len(input_symbol_vocab),config["conv1_knum"], kernel_size=config["conv1_ksize"], stride=1, padding=config["conv1_ksize"]//2)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=config["pool1"], stride=config["pool1"])
        self.activation1 = self.get_activation(config["conv_activation"])
        self.conv1dropout = torch.nn.Dropout(config["conv_dropout"])
        
        self.conv2 = torch.nn.Conv1d(config["conv1_knum"],config["conv2_knum"], kernel_size=config["conv2_ksize"], stride=1, padding=config["conv2_ksize"]//2)
        self.pool2 = torch.nn.MaxPool1d(kernel_size=config["pool2"], stride=config["pool2"])
        self.activation2 = self.get_activation(config["conv_activation"])
        self.conv2dropout = torch.nn.Dropout(config["conv_dropout"])
        
        self.fc_t = torch.nn.Linear(325,self.xt_hidden) #assuming 325 TFs
        self.fc_t_activation = self.get_activation(config["fc_activation"])
        self.fc_t_dropout = torch.nn.Dropout(config["fc_dropout"])
        
        self.fc1 = torch.nn.Linear(config["conv2_knum"]*self.xs_hidden+self.xt_hidden, 512)
        self.bn1 = torch.nn.BatchNorm1d(num_features=512)
        self.fc1_activation = self.get_activation(config["fc_activation"])
        self.fc1_dropout = torch.nn.Dropout(config["fc_dropout"])
        
        self.fc2 = torch.nn.Linear(512, 256)
        self.bn2 = torch.nn.BatchNorm1d(num_features=256)
        self.fc2_activation = self.get_activation(config["fc_activation"])
        self.fc2_dropout = torch.nn.Dropout(config["fc_dropout"])
        
        self.fc3 = torch.nn.Linear(256, 64)
        self.bn3 = torch.nn.BatchNorm1d(num_features=64)
        self.fc3_activation = self.get_activation(config["fc_activation"])
        self.fc3_dropout = torch.nn.Dropout(config["fc_dropout"])
        
        self.fc_out = torch.nn.Linear(64,1)
        
    @classmethod
    def get_activation(cls, activation):
        if activation == "relu":
            return torch.nn.ReLU()
        elif activation == "gelu":
            return torch.nn.GELU()
        elif activation == "elu":
            return torch.nn.ELU()
        elif activation == "selu":
            return torch.nn.SELU()
        else:
            raise ValueError("Invalid activation function: must be relu, gelu, or elu.")
    
    def get_input_vocab(self):
        return self.input_symbol_vocab
    
    def forward(self, x_s, x_t):
        b=list(x_s.size())[0] #batch size
        x_s = self.conv1(x_s)
        x_s = self.pool1(x_s)
        x_s = self.activation1(x_s)
        x_s = self.conv1dropout(x_s)
        
        x_s = self.conv2(x_s)
        x_s = self.pool2(x_s)
        x_s = self.activation2(x_s)
        x_s = self.conv2dropout(x_s)
        
        x_t = self.fc_t(x_t)
        x_t = self.fc_t_activation(x_t)
        x_t = self.fc_t_dropout(x_t)
        
        x = torch.cat((x_s.reshape((b,self.config["conv2_knum"]*self.xs_hidden,1)).view(-1, self.config["conv2_knum"]*self.xs_hidden),x_t),1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc1_activation(x)
        x = self.fc1_dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.fc2_activation(x)
        x = self.fc2_dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.fc3_activation(x)
        x = self.fc3_dropout(x)
        
        x = self.fc_out(x)
        #print(self.convs[0].weight.data)
        #yay!
        return x    
    
class Trainable(tune.Trainable):
    
    def setup(self, config, output_classes, char_vocab, category_vocab, trainset, devset):
        """
        Initializes a trainer to train a neural network using the provided DataLoaders 
        (for training and development data).

        tensorize is a function that maps a piece of data (i.e. a dictionary)
        to an input and output tensor for the neural network.
        
        char_vocab is the Vocab object of the input data.

        """
        print("Setting up Trainable")
        self.config = config
        self.scaler = scaler()
        self.tensorize = Tensorize(char_vocab, config["seq_length"], category_vocab)
        self.char_vocab = char_vocab
        self.train_loader = DataLoader(trainset, batch_size=config["batch_size"], num_workers=4, shuffle=True)
        self.dev_loader = DataLoader(devset, batch_size=config["batch_size"], num_workers=4, shuffle=False)
            
        self.net = cudaify(ProCNN(config, output_classes, char_vocab))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config["learning_rate"])
        
    @classmethod
    def objective(cls):
        return torch.nn.MSELoss()
    
    @classmethod
    def correlation(self, x, y):
        return pearsonr(x, y)[0]
    
    def step(self):
        self.net.train()
        train_loss = self.run_one_cycle(True)
        self.net.eval()
        dev_loss, dev_corr = self.run_one_cycle(False)
        
        return {"train_loss" : train_loss,
                "dev_loss" : dev_loss,
                "dev_corr" : dev_corr
        }
    
    def run_one_cycle(self, train=True):
        total_loss = 0
        
        if not train:
            all_labels = []
            all_preds = []
        
        for i, data in enumerate(self.train_loader if train else self.dev_loader):
            print(i, "/", len(self.train_loader if train else self.dev_loader))
            if train:
                self.optimizer.zero_grad()
        
            input_s, input_t, labels = self.tensorize(data)
            outputs = self.net(input_s, input_t).view(-1)
            if use_cuda:
                with torch.cuda.amp.autocast():
                    loss = self.objective()(outputs, labels)
            else:
                loss = self.objective()(outputs, labels)

            total_loss += loss.data.item()
            
            if not train:
                all_labels.extend(labels.tolist())
                all_preds.extend(outputs.tolist())
            
            if train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        total_loss = total_loss / len(self.train_loader if train else self.dev_loader)
        if not train:
            total_corr = self.correlation(all_labels, all_preds)
            return total_loss, total_corr
    
        return total_loss

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint.pt")
        
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)
        
    def load_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.net.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])        
            

def search_hyperparameters(experiment_name, search_space, output_classes, trainset, devset):
    """
    Performs a hyperparameter search of the best-performing network
    for the given datasets (training and development).
    """   
    
    def extract_amino_acid_alphabet(trainset):
        #symbols = set()
        #for sequence in trainset.select('seq'):
        #    letters = set(sequence)
        #    symbols = symbols | letters
        #return sorted(list(symbols))
        return ['A', 'C', 'G', 'N', 'T']

    char_vocab = Vocab(extract_amino_acid_alphabet(trainset))    
    categories = sorted(list(set(trainset.select('exp'))))
    category_vocab = Vocab(categories)

    bayesopt = SkOptSearch(metric="dev_corr", mode="max")
    
    # Early stopper for plateauing metrics
    stopper = ray.tune.stopper.TrialPlateauStopper("dev_corr", std=0.01, num_results=5, grace_period=5)
    
    # https://arxiv.org/pdf/1810.05934.pdf
    scheduler = tune.schedulers.ASHAScheduler(
        time_attr='training_iteration',
        metric='dev_corr',
        mode='max',
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1
    )

    print("Starting search")
    results = tune.run(tune.with_parameters(Trainable, 
                                            char_vocab=char_vocab,
                                            category_vocab=category_vocab,
                                            output_classes=output_classes,
                                            trainset=trainset,
                                            devset=devset),
                        config=search_space,
                        name=experiment_name,
                        resources_per_trial={"cpu" : 8, "gpu" : 0.5},
                        num_samples=500,
                        search_alg=bayesopt,
                        scheduler=scheduler,
                        stop=stopper,
                        max_concurrent_trials=2,
                        fail_fast=True)
    
    return results


if __name__ == "__main__":
    search_space = {
       "learning_rate": tune.loguniform(1e-5, 1e-1),
       "xt_hidden": tune.choice([16, 32, 64, 128, 256, 512, 1024]),
       "seq_length": tune.choice([i * 100 for i in range(1, 11)]),
       "conv1_ksize": tune.choice([7, 9, 11, 13, 15]),
       "conv1_knum": tune.choice([16, 32, 64, 128, 256]),
       "pool1": tune.randint(1,20),
       "conv2_ksize": tune.choice([7, 9, 11, 13, 15]),
       "conv2_knum": tune.choice([16, 32, 64, 128, 256]),
       "pool2": tune.randint(1,20),
       "batch_size": tune.choice([2, 4, 8, 16, 32, 64, 128, 256]),
       "conv_activation" : tune.choice(["relu", "gelu", "elu", "selu"]),
       "fc_activation" : tune.choice(["relu", "gelu", "elu", "selu"]),
       "conv_dropout": tune.uniform(0, 0.50),
       "tf_dropout": tune.uniform(0, 0.50),
       "fc_dropout": tune.uniform(0, 0.50),
    }


    print("Loading data")
    trainset = JsonDataset('/home/simonl2/yeast/Gene_Expression_Pred/October_Runs/Data/file_oav_filt05_filtcv3_Zlog_new_trainGenes.json')
    devset = JsonDataset('/home/simonl2/yeast/Gene_Expression_Pred/October_Runs/Data/file_oav_filt05_filtcv3_Zlog_new_validGenes.json')
    print("Loaded data")

    ray.init(
        num_cpus = 16,
        object_store_memory=10 * 1024 * 1024 * 1024,
        _redis_max_memory=3*1024*1024*1024,
    )

    results = search_hyperparameters("2021-11-09", search_space, 1, trainset, devset)
    
    best_trial = result.best_trial
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["val_loss"]))
    print("Best trial final validation correlation: {}".format(
        best_trial.last_result["val_corr"]))
    print(results.results_df)

