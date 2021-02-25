import time
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from data import JsonDataset, get_samplers
from data import domain, split_data
from torch.utils.data.sampler import SubsetRandomSampler
from copy import deepcopy
import pandas as pd

if False:
    print("using gpu")
    cuda = torch.device('cuda')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    def cudaify(model):
        return model.cuda()
else: 
    print("using cpu")
    cuda = torch.device('cpu')
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    def cudaify(model):
        return model
    
UNITS_PER_SIZE = 50

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
        category = LongTensor([self.category_vocab(c) 
                             for c in data['exp']])
        
        return cudaify(words), cudaify(tfs), cudaify(category)
        
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
            for li, letter in enumerate(word):
                tensor[i][vocab(letter)][li] = 1
        return tensor

class ProCNN(torch.nn.Module):
    """
    A two layer CNN that uses max pooling to extract information from
    the sequence input, a one layer fully-connected network to extract TF information 
    then trains a fully connected neural network on the to make predictions.
    
    """    
    def __init__(self, n_input_symbols, hidden_size,
                 input_symbol_vocab, output_classes):
        super(ProCNN, self).__init__()
        
        self.input_symbol_vocab = input_symbol_vocab
        
        self.hidden_size=hidden_size
        
        self.output_classes = output_classes
        
        self.conv1 = torch.nn.Conv1d(n_input_symbols,128, kernel_size=5, stride=1, padding=2)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=20, stride=20)
        self.conv2 = torch.nn.Conv1d(128,32, kernel_size=9, stride=1, padding=4)
        self.pool2 = torch.nn.MaxPool1d(kernel_size=10, stride=10)
        
        self.fc_t = torch.nn.Linear(325,hidden_size) #assuming 325 TFs
        
        self.fc1 = torch.nn.Linear(32*5+hidden_size, 512)
        self.bn1 = torch.nn.BatchNorm1d(num_features=512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 64)
        self.fc_out = torch.nn.Linear(64,output_classes)
    
    def get_input_vocab(self):
        return self.input_symbol_vocab
    
    def get_hidden(self, x):
        pass
    
    def forward(self, x_s, x_t):
        b=list(x_s.size())[0] #batch size
        x_s = self.conv1(x_s)
        x_s = self.pool1(x_s)
        x_s = F.relu(x_s)
        x_s = self.conv2(x_s)
        x_s = self.pool2(x_s)
        x_s = F.relu(x_s)
        
        x_t = F.relu(self.fc_t(x_t))
        
        x = torch.cat((x_s.reshape((b,32*5,1)).view(-1, 32*5),x_t),1)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc_out(x)
        #print(self.convs[0].weight.data)
        #yay!
        return x

class CnnClassifier:
    """
    Wraps a trained neural network to use for classification.
    
    """    
    def __init__(self, cnn, tensorize):
        self.net = cnn
        self.tensorize = tensorize
     
    def run(self, data_loader):
        for data in data_loader:
            inputs, _ = self.tensorize(data)
            yield self.net(inputs)  
            
    def evaluate(self, data_loader):
        """
        To fix
        """
        correct = 0
        total = 0
        confusion_mat=[[0 for j in range(self.net.output_classes)]for i in range(self.net.output_classes)]
        for data in data_loader:
            input_s, input_t, labels = self.tensorize(data)
            outputs = self.net(input_s,input_t)
            correct_inc, total_inc = CnnClassifier.accuracy(outputs, labels)
            confusion_mat_inc = CnnClassifier.confusion(outputs, labels)
            correct += correct_inc
            total += total_inc
            for i in range(len(confusion_mat)):
                for j in range(len(confusion_mat)):
                    confusion_mat[i][j]+=confusion_mat_inc[i][j]
        return correct / total, confusion_mat
    
    @staticmethod
    def accuracy(outputs, labels):
        correct = 0
        total = 0
        for output, label in zip(outputs, labels):
            total += 1
            if label.item() == output.argmax().item():
                correct += 1
        return correct, total
    
    @staticmethod
    def confusion(outputs,labels):
        mat=[[0 for j in range(len(outputs[0]))]for i in range(len(outputs[0]))]
        for output, label in zip(outputs,labels):
            i = output.argmax().item()
            j=label.item()
            mat[i][j]+=1
        return mat

def train_network(net, tensorize, 
                  train_loader, dev_loader, 
                  n_epochs, learning_rate):
    """
    Trains a neural network using the provided DataLoaders (for training
    and development data).
    
    tensorize is a function that maps a piece of data (i.e. a dictionary)
    to an input and output tensor for the neural network.
    
    """
    print("===== HYPERPARAMETERS =====")
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    n_batches = len(train_loader)
    print_every = n_batches // 10  
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    training_start_time = time.time()
    best_accuracy = 0.0
    best_net = None
    for epoch in range(n_epochs):
        running_loss = 0.0
        start_time = time.time()
        total_train_loss = 0
        for i, data in enumerate(train_loader, 0):
            input_s, input_t, labels = tensorize(data)
            optimizer.zero_grad()
            outputs = net(input_s,input_t)
            #print('output: ',outputs)
            #print('label: ',labels)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), 
                        running_loss / print_every, time.time() - start_time))
                #print(outputs)
                #print(labels)
                running_loss = 0.0
                start_time = time.time()                    
        #At the end of the epoch, do a pass on the validation set
        accuracy, confusion_mat = CnnClassifier(net, tensorize).evaluate(dev_loader)
        print(accuracy)
        if accuracy > best_accuracy:
            best_confusion = confusion_mat
            print(best_confusion)
            best_net = deepcopy(net)
            best_accuracy=accuracy
            torch.save(best_net,'current_best_f_cv5_lt.nnt')
        print("Dev acc = {:.2f}".format(accuracy))
        #print(outputs)
        #print(labels)
               
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    print("Best mse: ", best_mse)
    return best_net

def train_all_categories(dataset, hidden_size=512):
    """
    Trains a multi-way CNN classifier for all protein families in the
    dataset.
    
    e.g. train_all_categories(JsonDataset('data/var7.10.json'))
    
    """   
    def extract_amino_acid_alphabet(dataset):
        symbols = set()
        for sequence in dataset.select('seq'):
            letters = set(sequence)
            symbols = symbols | letters
        return sorted(list(symbols))

    train_sampler, val_sampler, _ = get_samplers(range(len(dataset)), 
                                                 .25, .05)
    char_vocab = Vocab(extract_amino_acid_alphabet(dataset))
    
    categories = sorted(list(set(dataset.select('exp'))))
    category_vocab = Vocab(categories)
    print(category_vocab.index_map)
    CNN = cudaify(ProCNN(len(char_vocab.alphabet),
                            hidden_size,
                            char_vocab, len(category_vocab)))
    train_loader =  DataLoader(dataset, batch_size=125,
                               sampler=train_sampler, num_workers=0)
    dev_loader =  DataLoader(dataset, batch_size=64,
                             sampler=val_sampler, num_workers=0)
    bestCNN=train_network(CNN, Tensorize(char_vocab, 1024, category_vocab), 
                  train_loader, dev_loader, n_epochs=30, learning_rate=0.0001)
    return bestCNN


if __name__ == "__main__":
   train_all_categories(JsonDataset('test_file_oav_filt05_filtcv3_Zlog_tri.json'))

