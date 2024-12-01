import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.utils.data import Dataset, DataLoader
import time

class bioSpikeDatasetClass(Dataset):
    def __init__(self, spike_data, overlap=100, sequence_length=300):

        # Convert data to tensor
        self.data = torch.FloatTensor(spike_data)
        self.sequence_length = sequence_length
        
        # Create sequences with overlap
        self.sequences = torch.stack([
            self.data[i:i + sequence_length] 
            for i in range(0, len(self.data) - sequence_length, overlap)])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

class SpikingGenerator(nn.Module):
    def __init__(self, num_neurons, num_hidden):
        super().__init__()

        self.num_neurons = num_neurons
        self.num_hidden = num_hidden

        # Fully Conn and LIF layers
        self.fc1 = nn.Linear(num_neurons, num_hidden)
        self.lif1 = snn.Leaky(beta=1)
        self.fc2 = nn.Linear(num_hidden, num_neurons)
        self.lif2 = snn.Leaky(beta=1)

        # Recurrent connection for memory
        self.fc_recurrent = nn.Linear(num_hidden, num_hidden)

    def forward(self, x, num_steps, live_plot=False):
        batch_size = x.shape[0]

        # Initialize hidden state and membrane potential
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1_last = torch.zeros(batch_size, self.fc1.out_features)
        
        # For live plotting
        fig = None
        if live_plot:
            plt.ion()
            fig, ax = plt.subplots(figsize=(12, 6))
            spike_times = [[] for _ in range(self.num_neurons)]
            spike_neurons = [[] for _ in range(self.num_neurons)]
        
        # Process initial input
        cur1 = self.fc1(x) + self.fc_recurrent(spk1_last)
        spk1, mem1 = self.lif1(cur1, mem1)
        spk1_last = spk1
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        current_spikes = spk2

        if live_plot:
            spikes = current_spikes.detach().numpy().squeeze()
            for n in range(self.num_neurons):
                if spikes[n] > 0.7:
                    spike_times[n].append(0)
                    spike_neurons[n].append(n)
        
        all_spikes = [current_spikes]
        
        # Generate remaining sequence
        for t in range(1, num_steps):
            cur1 = self.fc1(current_spikes) + self.fc_recurrent(spk1_last)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_last = spk1
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            current_spikes = spk2
            all_spikes.append(current_spikes)
            
            if live_plot:
                spikes = current_spikes.detach().numpy().squeeze()
                for n in range(self.num_neurons):
                    if spikes[n] > 0.5:
                        spike_times[n].append(t)
                        spike_neurons[n].append(n)
                
                if t % 5 == 0:  # Update plot every 5 steps
                    ax.clear()
                    for n in range(self.num_neurons):
                        if len(spike_times[n]) > 0:
                            ax.scatter(spike_times[n], spike_neurons[n], 
                                     marker='|', color='black', s=10)
                    ax.set_title(f'Spike Raster Plot (Step {t + 5}/{num_steps})')
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Neuron')
                    ax.set_ylim(-0.5, self.num_neurons-0.5)
                    ax.set_xlim(-1, num_steps)
                    plt.pause(0.01)
        
        if live_plot:
            plt.ioff()
            return spike_times, spike_neurons, fig
        
        return torch.stack(all_spikes, dim=1)

def train_model(model, train_loader, num_epochs=2):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print("Training progress:")
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Get initial state and target sequence
            initial_state = batch[:, 0, :]
            target_sequence = batch[:, 1:, :]
            
            # Forward pass
            output = model(initial_state, num_steps=target_sequence.shape[1])
            
            # Calculate loss
            loss = criterion(output, target_sequence)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"\rEpoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}", end="")
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss/len(train_loader):.4f}")


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

# Load data
fname = "data.npz"  # Your data file
data = np.load(fname, allow_pickle=True)
spike_train = data['train'].item()
sample_freq = data['fs']

# Convert spike train to binned data
binned_data = convert_to_binned(spike_train, sample_freq)

# Plot binned data
plt.figure(figsize=(14, 6))
plt.title("Binned Spike Train Data")
plt.xlabel("Time (ms)")
plt.ylabel("Neuron")
y = 0
for i in range(binned_data.shape[1]):
    plt.eventplot(np.where(binned_data[:, i] > 0)[0], lineoffsets=y, linelengths=0.5, colors='black')
    y += 1

# Optionally plot the total population firing rate over time
plt.plot(np.sum(binned_data, axis=1), color='black', linewidth=2, alpha=0.2)
plt.show()


# num_neurons = binned_data.shape[1]
# num_hidden = 500
# time_steps = 2000

# # Create dataset and loader
# dataset = bioSpikeDatasetClass(binned_data)
# train_loader = DataLoader(dataset, batch_size=500, shuffle=False)

# # Create and train network
# net = SpikingGenerator(num_neurons, num_hidden)
# print("Training network...")
# train_model(net, train_loader)

# # Generate new sequence with live plotting
# print("\nGenerating new sequence...")
# initial_state = torch.zeros(1, num_neurons)
# initial_state[0, num_neurons//2] = 1  # Start with middle neuron firing

# with torch.no_grad():
#     spike_times, spike_neurons, fig = net(initial_state, time_steps, live_plot=True)

# plt.show()