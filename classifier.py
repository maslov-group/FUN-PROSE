import argparse
import os
import pandas as pd
from scipy.stats import pearsonr
import torch
from torch.utils.data import DataLoader

from data import PandasDataset

torch.backends.cudnn.benchmark = True


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
    
    def __init__(self, symbol_vocab, max_word_length, cudaify):
        self.symbol_vocab = symbol_vocab
        self.max_word_length = max_word_length
        self.cudaify = cudaify
    
    def __call__(self, data):
        words = Tensorize.words_to_tensor(data['seq'], 
                                              self.symbol_vocab, 
                                              self.max_word_length).float()
        
        tfs = torch.stack(data['tfs'],dim=1).float()
        label = data['exp'].float()
        return self.cudaify(words), self.cudaify(tfs), self.cudaify(label)
        
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
            start_index = max(0, len(word) - max_word_length)
            for li, letter in enumerate(word[start_index:len(word)][::-1]):
                tensor[i][vocab(letter)][max_word_length - li - 1] = 1
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
        self.conv1bn = torch.nn.BatchNorm1d(config["conv1_knum"])
        
        self.conv2 = torch.nn.Conv1d(config["conv1_knum"],config["conv2_knum"], kernel_size=config["conv2_ksize"], stride=1, padding=config["conv2_ksize"]//2)
        self.pool2 = torch.nn.MaxPool1d(kernel_size=config["pool2"], stride=config["pool2"])
        self.activation2 = self.get_activation(config["conv_activation"])
        self.conv2dropout = torch.nn.Dropout(config["conv_dropout"])
        self.conv2bn = torch.nn.BatchNorm1d(config["conv2_knum"])
        
        self.fc_t = torch.nn.Linear(config["num_tfs"],self.xt_hidden)
        self.fc_t_activation = self.get_activation(config["fc_activation"])
        self.fc_t_dropout = torch.nn.Dropout(config["fc_dropout"])
        
        self.fc1 = torch.nn.Linear(config["conv2_knum"]*self.xs_hidden+self.xt_hidden, 512)
        self.fc1_activation = self.get_activation(config["fc_activation"])
        self.fc1_dropout = torch.nn.Dropout(config["fc_dropout"])
        self.bn1 = torch.nn.BatchNorm1d(num_features=512)
        
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc2_activation = self.get_activation(config["fc_activation"])
        self.fc2_dropout = torch.nn.Dropout(config["fc_dropout"])
        self.bn2 = torch.nn.BatchNorm1d(num_features=256)
        
        self.fc3 = torch.nn.Linear(256, 64)
        self.fc3_activation = self.get_activation(config["fc_activation"])
        self.fc3_dropout = torch.nn.Dropout(config["fc_dropout"])
        self.bn3 = torch.nn.BatchNorm1d(num_features=64)
        
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
        x_s = self.conv1bn(x_s)
        
        x_s = self.conv2(x_s)
        x_s = self.pool2(x_s)
        x_s = self.activation2(x_s)
        x_s = self.conv2dropout(x_s)
        x_s = self.conv2bn(x_s)
        
        x_t = self.fc_t(x_t)
        x_t = self.fc_t_activation(x_t)
        x_t = self.fc_t_dropout(x_t)
        
        x = torch.cat((x_s.reshape((b,self.config["conv2_knum"]*self.xs_hidden,1)).view(-1, self.config["conv2_knum"]*self.xs_hidden),x_t),1)
        x = self.fc1(x)
        x = self.fc1_activation(x)
        x = self.fc1_dropout(x)
        x = self.bn1(x)
        
        x = self.fc2(x)
        x = self.fc2_activation(x)
        x = self.fc2_dropout(x)
        x = self.bn2(x)
        
        x = self.fc3(x)
        x = self.fc3_activation(x)
        x = self.fc3_dropout(x)
        x = self.bn3(x)
        
        x = self.fc_out(x)
        return x
    
class Trainable():
    
    class MockScaler():
        def scale(self, x):
            return x
        def step(self, optim):
            optim.step()
        def update(self):
            pass
        
    def setup(self, config, output_classes, char_vocab, trainset, devset):
        """
        Initializes a trainer to train a neural network using the provided DataLoaders 
        (for training and development data).

        tensorize is a function that maps a piece of data (i.e. a dictionary)
        to an input and output tensor for the neural network.
        
        char_vocab is the Vocab object of the input data.

        """
        print("Setting up Trainable")
        self.config = config
        self.char_vocab = char_vocab
        self.train_loader = DataLoader(trainset, batch_size=config["batch_size"], num_workers=4, pin_memory=True, shuffle=True)
        self.dev_loader = DataLoader(devset, batch_size=config["batch_size"], num_workers=4, pin_memory=True, shuffle=False)

        self.best_dev_corr = 0
        
        self.use_cuda = config["use_cuda"]
        if self.use_cuda:
            self.cudaify_model = lambda x: x.cuda()
            self.cudaify = lambda x: x.cuda(non_blocking=True)
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.cudaify_model = lambda x: x
            self.cudaify = lambda x: x
            self.scaler = MockScaler()
        
        self.tensorize = Tensorize(char_vocab, config["seq_length"], self.cudaify)
        self.net = self.cudaify_model(ProCNN(config, output_classes, char_vocab))
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        self.loss = torch.nn.MSELoss()

        self.terminate = False

    def cleanup(self):
        self.terminate = True

        del self.config
        del self.tensorize
        del self.char_vocab
        del self.train_loader
        del self.dev_loader

        del self.best_dev_corr
        
        del self.use_cuda
        del self.cudaify_model
        del self.cudaify
        del self.scaler

        del self.net
        del self.optimizer
        del self.loss

        torch.cuda.empty_cache()

        
    @classmethod
    def correlation(self, x, y):
        return pearsonr(x, y)[0]
    
    def step(self):
        self.net.train()
        train_loss = self.run_one_cycle(train=True)
        self.net.eval()
        with torch.no_grad():
            dev_loss, dev_corr = self.run_one_cycle(train=False)
            self.best_dev_corr = max(self.best_dev_corr, dev_corr)
        
        return {
                "train_loss" : train_loss,
                "dev_loss" : dev_loss,
                "dev_corr" : dev_corr,
                "best_dev_corr" : self.best_dev_corr,
        }
    
    def run_one_cycle(self, train=True):
        total_loss = 0
        
        if not train:
            all_labels = []
            all_preds = []
        
        for i, data in enumerate(self.train_loader if train else self.dev_loader):
            if self.terminate:
                return None
            if i % 10 == 0:
                print(i, "/", len(self.train_loader if train else self.dev_loader))
            if train:
                self.optimizer.zero_grad()
        
            input_s, input_t, labels = self.tensorize(data)
            outputs = self.net(input_s, input_t).view(-1)
            if self.use_cuda:
                with torch.cuda.amp.autocast():
                    loss = self.loss(outputs, labels)
            else:
                loss = self.loss(outputs, labels)
            
            if not train:
                all_labels.extend(labels.detach().tolist())
                all_preds.extend(outputs.detach().tolist())
            
            if train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss += loss.detach().item()


        total_loss = total_loss / len(self.train_loader if train else self.dev_loader)
        if not train:
            total_corr = self.correlation(all_labels, all_preds)
            return total_loss, total_corr
    
        return total_loss

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint.pt")
        
        torch.save({
            "best_dev_corr" : self.best_dev_corr,
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, path)

        return checkpoint_dir


        
    def load_checkpoint(self, checkpoint_path):
        path = os.path.join(checkpoint_path, "checkpoint.pt")
        state = torch.load(path)

        self.best_dev_corr = state["best_dev_corr"]
        self.net.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])        
            

def run_single_model(args, mode_config, output_classes, trainset, devset):
    """
    Performs a hyperparameter search of the best-performing network
    for the given datasets (training and development).
    """   
    
    def extract_amino_acid_alphabet(trainset):
        return ['A', 'C', 'G', 'N', 'T']

    char_vocab = Vocab(extract_amino_acid_alphabet(trainset))    

    trainer = Trainable()
    trainer.setup(config=config,char_vocab=char_vocab, output_classes=output_classes, trainset=trainset, devset=devset)
    
    epoch = 1
    while True:
        print(f"Epoch {epoch} ", trainer.step())
        epoch += 1
    
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run single classifier for FUN-PROSE")
    parser.add_argument("-use_cuda", type=bool, default=False)
    args = parser.parse_args()

    config = {
       "learning_rate": 0.00010229218879330196,
       "weight_decay": 0.0016447149582678627,
       "xt_hidden": 1024,
       "seq_length": 1000,
       "conv1_ksize": 9,
       "conv1_knum": 256,
       "pool1": 19,
       "conv2_ksize": 13,
       "conv2_knum": 64,
       "pool2": 8,
       "batch_size": 256,
       "conv_activation" : "relu",
       "fc_activation" : "elu",
       "conv_dropout": 0.3981796388676127,
       "tf_dropout": 0.18859739941162465,
       "fc_dropout": 0.016570328292903613,
       "use_cuda": args.use_cuda,
       "num_tfs":325
    }


    print("Loading data")
    trainset = PandasDataset('/home/simonl2/yeast/Gene_Expression_Pred/October_Runs/Data/file_oav_filt05_filtcv3_Zlog_new_trainGenes.pkl')
    devset = PandasDataset('/home/simonl2/yeast/Gene_Expression_Pred/October_Runs/Data/file_oav_filt05_filtcv3_Zlog_new_validGenes.pkl')
    print("Loaded data")

    run_single_model(args, config, 1, trainset, devset)

