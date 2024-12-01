import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class SpikeTrainDataset(Dataset):

    ''' 
    Dataset of spike trains for testing 
    
    Note that len and getitem methods are required for PyTorch datasets
    '''

    def __init__(self, num_samples=1000, num_neurons=100, time_steps=50, firing_rate=0.1):
        self.data = []
        for _ in range(num_samples):
            spike_train = np.random.binomial(1, firing_rate, (time_steps, num_neurons))
            self.data.append(torch.FloatTensor(spike_train))
        self.data = torch.stack(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SimpleAutoencoder(nn.Module):
    '''
    Autoencoder with a single linear layer in the encoder and decoder.

    The encoder maps input data to a latent space representation, 
    and the decoder maps the latent space representation back to 
    the original input space. Generative model.
    '''
    def __init__(self, input_dim, latent_dim, threshold=0.18):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.threshold = threshold
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        latent = self.encoder(x)
        reconstruction = self.sigmoid(self.decoder(latent))
        if not self.training:
            reconstruction = (reconstruction > self.threshold).float()
        return reconstruction, latent
    
class RealSpikeDataset(Dataset):
    def __init__(self, data, chunk_size=1000, n_neurons=None):
        self.train = data['train'].item()
        neuron_ids = sorted(self.train.keys())[:n_neurons] if n_neurons else sorted(self.train.keys())
        self.train = {k: self.train[k] for k in neuron_ids}
        
        max_time = max(max(spikes) for spikes in self.train.values())
        self.chunks = []
        
        for start in range(0, int(max_time), chunk_size):
            chunk = torch.zeros((chunk_size, len(neuron_ids)))
            for nid, spikes in self.train.items():
                idx = neuron_ids.index(nid)
                mask = (spikes >= start) & (spikes < start + chunk_size)
                if any(mask):
                    times = spikes[mask] - start
                    chunk[times.astype(int), idx] = 1
            self.chunks.append(chunk.unsqueeze(0))
            
        self.data = self.chunks[0] # Use first chunk by default
        
    def __len__(self):
        return len(self.chunks)
        
    def __getitem__(self, idx):
        return self.chunks[idx]

def train_simple_autoencoder(model, train_loader, epochs=1):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            batch = batch.view(batch.size(0), -1)
            optimizer.zero_grad()
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}')


def sample_to_tensor(sample):
    return sample[0].reshape(dataset.time_steps, dataset.num_neurons)


def tensor_to_spike_times(tensor, freq=20000):
    spike_times = []
    for neuron_idx in range(tensor.size(1)):
        spike_indices = torch.nonzero(tensor[:, neuron_idx]).squeeze()
        spike_times.append(spike_indices.float()/freq)
    return spike_times

def plot_spike_times(spike_times):
    plt.figure(figsize=(10, 5))
    for neuron_idx, spikes in enumerate(spike_times):
        plt.plot(spikes, torch.ones_like(spikes)*neuron_idx, '.', markersize=5, alpha=0.5, color='blue')  
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Index')
    plt.title('Spike Times')
    plt.show()

# class SpikeDataset(Dataset):
#     def __init__(self, data, max_time=10000):
#         self.freq = data['fs']
#         self.train = data['train'].item()
#         self.neuron_ids = sorted(self.train.keys())
#         self.id_to_idx = {id: idx for idx, id in enumerate(self.neuron_ids)}
#         self.num_neurons = len(self.neuron_ids)
#         self.dt = 1
#         self.max_time = min(max(max(v) for v in self.train.values()), 
#             max_time) if max_time else max(max(v) for v in self.train.values())
#         self.time_steps = int(self.max_time/self.dt)
#         self.data = self.TrainToTensor().unsqueeze(0)
        
#     def TrainToTensor(self):
#         spike_tensor = torch.zeros((self.time_steps, self.num_neurons))
#         for neuron_id, spike_times in self.train.items():
#             indices = torch.floor(torch.tensor(spike_times)).long()
#             valid_indices = indices[indices < self.time_steps]
#             spike_tensor[valid_indices, self.id_to_idx[neuron_id]] = 1
#         return spike_tensor
        
#     def getTrain(self):
#         return self.train
        
#     def __len__(self):
#         return 1
        
#     def __getitem__(self, idx):
#         return self.data[idx]
        
#     def getShape(self):
#         return self.data.shape

    
# fname = "data.npz"
# data = np.load(fname, allow_pickle=True)
# spike_data = SpikeDataset(data)

# num_neurons = spike_data.num_neurons
# time_steps = spike_data.time_steps
# input_dim = num_neurons * time_steps
# latent_dim = 200

# # bioTrainData = DataLoader(spike_data, batch_size=5, shuffle=True)
# # model = SimpleAutoencoder(input_dim, latent_dim)
# # train_simple_autoencoder(model, bioTrainData)
# # sample = next(iter(bioTrainData))   
# # reconstruction, latent = reconstruct_and_plot(model, sample, time_steps, num_neurons)

### Synthetic Data Example ###

num_neurons = 5
time_steps = 1000
input_dim = num_neurons * time_steps
latent_dim = 200

dataset = SpikeTrainDataset(num_neurons=num_neurons, time_steps=time_steps)
print(dataset[1].shape)

train_loader = DataLoader(dataset, batch_size=5, shuffle=True)

model = SimpleAutoencoder(input_dim, latent_dim)
train_simple_autoencoder(model, train_loader)

sample = next(iter(train_loader))




