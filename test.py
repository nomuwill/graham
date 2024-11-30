import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


class DictSpikeDataset(Dataset):
    def __init__(self, spike_dict, time_steps=50, time_bin_size=100000):
        self.data = []
        
        for neuron_id, spike_times in spike_dict.items():
            indices = (torch.tensor(spike_times) // time_bin_size).long()
            indices = indices[indices < time_steps]
            spike_train = torch.zeros(time_steps)
            spike_train[indices] = 1
            self.data.append(spike_train)
            
        self.data = torch.stack(self.data)
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.data
    
    def getShape(self):
        return self.data.shape




fname = "data.npz"
data = np.load(fname, allow_pickle=True)
sample_freq = data['fs']              
spike_train = data['train'].item()  

dataset = DictSpikeDataset(spike_train)
loader = DataLoader(dataset, batch_size=1)
print(dataset.getShape())

