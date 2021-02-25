from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import random
import json
import pandas as pd

class JsonDataset(Dataset):
    """
    Creates a Torch Dataset from a JSON file.
    
    We assume the JSON file is a list of dictionaries, where each
    dictionary corresponds to a single datum.
    
    """    
    def __init__(self, json_file):
        with open(json_file) as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def select(self, field):
        for datum in self:
            yield datum[field]

    

def domain(data, category):
    """
    Returns the set of existing values for a particular category in a 
    list of training data.
    
    """
    result = set()
    for datum in data:
        result.add(datum[category])
    return result


def split_data(ids, dev_percent, test_percent):
    """
    Given a list of datum ids and dev/test percentages, returns a partition
    (train, dev, test) of the datum ids.
    
    """
    dev_size = int(dev_percent * len(ids))
    test_size = int(test_percent * len(ids))
    train_ids = set(ids)
    dev_ids = random.sample(train_ids, dev_size)
    train_ids = train_ids - set(dev_ids)
    test_ids = random.sample(train_ids, test_size)
    train_ids = list(train_ids - set(test_ids))
    return train_ids, dev_ids, test_ids

def get_samplers(all_ids, dev_percent, test_percent):
    """
    Given a list of datum ids and dev/test percentages, makes a
    train/dev/test split and returns samplers for the three subsets.
    
    """        
    train_ids, dev_ids, test_ids = split_data(all_ids, dev_percent, test_percent)
    train_sampler = SubsetRandomSampler(train_ids)
    dev_sampler = SubsetRandomSampler(dev_ids)
    test_sampler = SubsetRandomSampler(test_ids)
    return train_sampler, dev_sampler, test_sampler    
    

