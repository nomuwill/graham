import snntorch as snn
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class SimpleGenSNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, beta=0.95, num_steps=100, sparsity=0.5):
        super().__init__()
        self.num_steps = num_steps
        self.sparsity = sparsity
        
        self.noise_gen = nn.Linear(input_size, input_size)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        # Add learnable parameters for thresholding
        self.threshold = nn.Parameter(torch.ones(1) * 0.5)
    
    def generate_initial_state(self, batch_size, device='cpu'):
        noise = torch.randn(batch_size, self.noise_gen.in_features, device=device, requires_grad=True)
        return torch.sigmoid(self.noise_gen(noise))
    
    def forward(self, x, mem_prev=None):
        batch_size = x.size(0)
        
        if mem_prev is None:
            mem_prev = torch.zeros(batch_size, self.fc1.out_features, device=x.device, requires_grad=True)
        
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem_prev)
        
        output = self.fc_out(spk1)
        # output = torch.sigmoid(output)  # Make output differentiable
        
        # Create sparse output with learnable threshold
        sparsity_mask = (torch.rand_like(output) < self.sparsity).float()
        output = output * sparsity_mask
        
        return output, mem1

    def generate(self, num_samples=1, device='cpu'):
        spike_trains = []
        
        current_state = self.generate_initial_state(num_samples, device)
        mem_state = None
        
        for _ in range(self.num_steps):
            spikes, mem_state = self.forward(current_state, mem_state)
            spike_trains.append(spikes)
            current_state = spikes
        
        return torch.stack(spike_trains, dim=1)
    


# Make a generate loop, then a forward loop
# The generate can be arbitrary, for fordawrds needs to be 

# For step in range(data.size(0))
#     data = data[step] 











    

def train_model(model, train_loader, num_epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                generated = model.generate(batch.size(0))
                loss = criterion(generated, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.6f}")



### Dummy Data ###

# Create sparse dummy data
num_samples = 100
sequence_length = 500
num_neurons = 100
sparsity = 0.05

dummy_data = np.random.choice(
    [0, 1], 
    size=(num_samples, sequence_length, num_neurons), 
    p=[1-sparsity, sparsity]
).astype(np.float32)

class BiologicalDataset(Dataset):
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

    
### ### ### ### ###

# Create dataset and loader
dataset = BiologicalDataset(dummy_data)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create model
model = SimpleGenSNN(
    input_size=num_neurons,
    hidden_size=64,
    output_size=num_neurons,
    num_steps=sequence_length,
    sparsity=sparsity
)

print("Starting training...")
train_model(model, train_loader)

print("Generating new data...")
with torch.no_grad():
    generated_data = model.generate(num_samples=5)

def plot_spike_trains(original, generated):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    ax1.imshow(original[0].T, aspect='auto', cmap='Blues')
    ax1.set_title('Original Spike Train')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Neuron')
    
    ax2.imshow(generated[0].numpy().T, aspect='auto', cmap='Blues')
    ax2.set_title('Generated Spike Train')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Neuron')
    
    plt.tight_layout()
    plt.show()

plot_spike_trains(dummy_data, generated_data)