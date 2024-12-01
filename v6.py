import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
from snntorch import functional as snnfunc
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class PatternGenerator(nn.Module):
    def __init__(self, num_input=182, num_hidden=256, beta=0.5, threshold=0.1):
        super().__init__()
        self.num_neurons = num_input
        self.num_hidden = num_hidden
        
        self.pattern_init = nn.Linear(1, num_input)
        self.fc1 = nn.Linear(num_input, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc2 = nn.Linear(num_hidden, num_input)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold)
        self.fc_recurrent = nn.Linear(num_input, num_input)
        
    def generate(self, num_steps=50, batch_size=1):

        # Generate initial pattern from random seed
        seed = torch.randn(batch_size, 1)
        current_pattern = torch.tanh(self.pattern_init(seed))
        
        membrane1 = self.lif1.init_leaky()
        membrane2 = self.lif2.init_leaky()
        
        patterns = []
        for step in range(num_steps):

            # Generate next pattern through SNN
            current1 = self.fc1(current_pattern)
            spike1, membrane1 = self.lif1(current1, membrane1)
            
            current2 = self.fc2(spike1)
            spike2, membrane2 = self.lif2(current2, membrane2)
            
            # Recurrent connection influences next pattern
            recurrent = self.fc_recurrent(spike2) * 5
            current_pattern = torch.tanh(recurrent + current_pattern)
            
            patterns.append(current_pattern)
            
        return torch.stack(patterns, dim=0)

    def forward(self, x):
        
        sequence = []
        membrane1 = self.lif1.init_leaky()
        membrane2 = self.lif2.init_leaky()
        
        for t in range(x.size(0)):
            current1 = self.fc1(x[t])
            spike1, membrane1 = self.lif1(current1, membrane1)
            
            current2 = self.fc2(spike1)
            spike2, membrane2 = self.lif2(current2, membrane2)
            
            sequence.append(spike2)
            
        return torch.stack(sequence, dim=0)
    

def train_pattern_generator(model, data_loader, num_epochs, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            batch = batch.unsqueeze(0)  # Add time dimension
            optimizer.zero_grad()
            output = model(batch)

            # MSE Loss between input and output
            loss = criterion(output, batch)  # Compare with input sequence

            target_pop = batch.mean(dim=2)  # Target population rate
            output_pop = output.mean(dim=2)  # Generated population rate
            
            # MSE Loss between target and output population rates
            pop_loss = criterion(output_pop, target_pop)
            
            # Sparse loss to encourage sparsity in the output   
            sparse_loss = torch.mean(torch.abs(output))

            max_rate_loss = torch.mean(torch.relu(output.mean(dim=0) - 0.3))

            # Combine losses
            total_loss = loss + sparse_loss + pop_loss + max_rate_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch}, Loss: {total_loss/len(data_loader)}')
            
    return model


def convert_to_binned(spike_data, sample_freq, bin_size_ms=50):
    num_neurons = len(spike_data)
    
    # Calculate the total time in seconds based on the max spike time and sample_freq
    max_time_sec = max(max(v) for v in spike_data.values()) / sample_freq  # convert max time from samples to seconds
    total_time_ms = max_time_sec * 1000  # convert total time to milliseconds
    
    # Number of bins corresponding to the total time and bin size in milliseconds
    bins_per_second = 1000 / bin_size_ms
    num_bins = int(np.ceil(total_time_ms / bin_size_ms))  # total number of bins

    # Initialize binned data array (time bins x neurons)
    binned_data = np.zeros((num_bins, num_neurons))

    # Fill in the binned data array
    for neuron_idx, (neuron, times) in enumerate(spike_data.items()):
        # Convert spike times from samples to time in milliseconds
        times_ms = np.array(times) * 1000 / sample_freq  # spike times in milliseconds
        bin_indices = (times_ms // bin_size_ms).astype(int)  # calculate bin indices
        
        # Remove any spike that falls outside the time range
        valid_indices = (bin_indices >= 0) & (bin_indices < num_bins)
        
        # Add spikes to the corresponding bins
        np.add.at(binned_data[:, neuron_idx], bin_indices[valid_indices], 1)

    return binned_data

class SpikeDataClass(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def prepare_data(data, batch_size=32):
    dataset = SpikeDataClass(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Load data
fname = "data.npz"  # Your data file
data = np.load(fname, allow_pickle=True)
spike_train = data['train'].item()
sample_freq = data['fs']

# Convert spike train to binned data
binned_data = convert_to_binned(spike_train, sample_freq)

# Usage
model = PatternGenerator(num_input=binned_data.shape[1], num_hidden=256)
data_loader = prepare_data(binned_data, batch_size=32)
model = train_pattern_generator(model, data_loader, num_epochs=10)
generated_data = model.generate(num_steps=binned_data.shape[0])

# Remove time dimension for plotting
generated_data = generated_data.squeeze().detach().numpy()

# Plot training data
plt.figure(figsize=(5, 10))
plt.subplot(2, 1, 1)  # First subplot for spike trains
plt.title("Spike Train Data")
plt.xlabel("Time")
plt.ylabel("Neuron")
y_offset = 0  # Offset for plotting each neuron's spikes
for i in range(binned_data.shape[1]):  # Iterate over neurons
    plt.eventplot(np.where(binned_data[:, i] > 0)[0], lineoffsets=y_offset, linelengths=0.1, colors='black', linewidths=0.5)
    y_offset += 1 
plt.plot(np.sum(binned_data, axis=1), color='black', linewidth=2, alpha=0.2)

# Plot generated spikes
plt.subplot(2, 1, 2)  # Second subplot for generated spikes
plt.title("Generated Spike Data")
plt.xlabel("Time")
plt.ylabel("Neuron")
y_offset = 0 
for i in range(generated_data.shape[1]):  # Iterate over neurons
    plt.eventplot(np.where(generated_data[:, i] > 0)[0], lineoffsets=y_offset, linelengths=0.1, colors='black')
    y_offset += 1
plt.plot(np.sum(generated_data, axis=1), color='black', linewidth=2, alpha=0.2)
plt.tight_layout()
plt.show()



