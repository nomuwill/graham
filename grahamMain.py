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
    def __init__(self, input_dim, latent_dim, threshold=0.3):
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

def train_simple_autoencoder(model, train_loader, epochs=10):
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

def reconstruct_and_plot(model, sample_data, time_steps, num_neurons): 

    ''' Create raster plot of original and reconstructed spike trains '''
    model.eval()
    with torch.no_grad():
        sample_flat = sample_data.view(sample_data.size(0), -1)
        reconstruction, latent = model(sample_flat)
        
    original = sample_data[0].reshape(time_steps, num_neurons)
    reconstructed = reconstruction[0].reshape(time_steps, num_neurons)
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 7))
    axs[0].imshow(original.T, cmap='Binary', aspect='auto')
    axs[0].set_title('Original Spike Train')
    axs[1].imshow(reconstructed.T, cmap='Binary', aspect='auto')
    axs[1].set_title('Reconstructed Spike Train')
    plt.show()
    
    return reconstruction, latent


class SpikeDataset(Dataset):
    def __init__(self, data):

        # Extract data
        self.freq = data['fs']
        self.train = data['train'].item()

        # Fix neuron IDs, may be missing some
        self.neuron_ids = sorted(self.train.keys())
        self.id_to_idx = {id: idx for idx, id in enumerate(self.neuron_ids)}
        self.num_neurons = len(self.neuron_ids)

        # Time steps
        self.time_bin_ms = 1
        self.max_time = max(max(v) for v in self.train.values())
        self.time_steps = int(self.max_time/self.time_bin_ms) + 1

        # Convert to tensor
        self.data = self.TrainToTensor()
        
    def TrainToTensor(self, time_limit=1000):
        time_steps = min(self.time_steps, time_limit) if time_limit else self.time_steps
        spike_tensor = torch.zeros((time_steps, self.num_neurons))
        for neuron_id, spike_times in self.train.items():
            indices = torch.floor(torch.tensor(spike_times)).long()
            valid_indices = indices[indices < time_steps]
            spike_tensor[valid_indices, self.id_to_idx[neuron_id]] = 1
        return spike_tensor.unsqueeze(0)
        
    def getTrain(self):
        return self.train
        
    def __len__(self):
        return 1
        
    def __getitem__(self, idx):
        return self.data[idx]
        
    def getShape(self):
        return self.data.shape

    
fname = "data.npz"
data = np.load(fname, allow_pickle=True)
spike_data = SpikeDataset(data)
spike_data.TrainToTensor()
print(spike_data.getShape())








# loader = DataLoader(dataset, batch_size=1)






# num_neurons = 50
# time_steps = 100
# input_dim = num_neurons * time_steps
# latent_dim = 200

# dataset = SpikeTrainDataset(num_neurons=num_neurons, time_steps=time_steps)
# print(dataset[1].shape)

# Shorten spike train
# print(spike_train)

# Create dataset

# train_loader = DataLoader(dataset, batch_size=5, shuffle=True)

# model = SimpleAutoencoder(input_dim, latent_dim)
# train_simple_autoencoder(model, train_loader)

# sample = next(iter(train_loader))
# reconstruction, latent = reconstruct_and_plot(model, sample, time_steps, num_neurons)