import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_neurons = 10
time_units = 200
n_initial_spikes = 5

# Create empty array
spike_data = np.zeros((time_units, n_neurons))

# Generate initial spike locations
all_positions = np.array([(t, n) for t in range(time_units) for n in range(n_neurons)])
initial_spikes = all_positions[np.random.choice(len(all_positions), n_initial_spikes, replace=False)]

# Function to add spreading activation
def add_spreading_activation(data, center_time, center_neuron, spread_time=3, spread_neurons=2):
    # Temporal spread (gaussian-like decay)
    for t in range(max(0, center_time - spread_time), min(time_units, center_time + spread_time + 1)):
        # Spatial spread (nearby neurons)
        for n in range(max(0, center_neuron - spread_neurons), min(n_neurons, center_neuron + spread_neurons + 1)):
            # Calculate distance-based probability
            time_dist = abs(t - center_time)
            neuron_dist = abs(n - center_neuron)
            if time_dist == 0 and neuron_dist == 0:
                continue  # Skip the center point as it's already set
            
            # Probability decreases with distance
            prob = 0.8 * np.exp(-0.5 * (time_dist + neuron_dist))
            if np.random.random() < prob:
                data[t, n] = 1

# Place initial spikes and add spreading activation
for pos in initial_spikes:
    spike_data[pos[0], pos[1]] = 1
    add_spreading_activation(spike_data, pos[0], pos[1])

# Visualize
plt.figure(figsize=(12, 6))
plt.imshow(spike_data.T, aspect='auto', cmap='binary')
plt.xlabel('Time')
plt.ylabel('Neuron')
plt.title('Dynamic Spike Train Data')
plt.colorbar(label='Spike')
plt.show()

print("Initial spike positions:")
for pos in initial_spikes:
    print(f"Time: {pos[0]}, Neuron: {pos[1]}")
print(f"\nTotal spikes: {int(np.sum(spike_data))}")