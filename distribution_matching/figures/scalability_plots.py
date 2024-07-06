import os
import pickle
import matplotlib.pyplot as plt
from scalability_analysis import methods

# Path to the folder containing the pickle files
folder_path = '../times'
os.makedirs('./plots', exist_ok=True)

# Initialize dictionaries for training and test times
train_times = {}
test_times = {}

# Load data from the pickle files
# for filename in os.listdir(folder_path):
for method, _ in methods():
    with open(os.path.join(folder_path, method+'.pkl'), 'rb') as file:
        (tr_times, te_times) = pickle.load(file)
        train_times[method] = tr_times
        test_times[method] = te_times

# Create the plot for training times
plt.figure(figsize=(10, 5))
for method, times in train_times.items():
    sizes = list(times.keys())
    time_values = list(times.values())
    plt.plot(sizes, time_values, marker='o', label=method)
plt.xlabel('Training dataset size')
plt.ylabel('Training time (s)')
plt.title('Training Times')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both")
plt.legend()
plt.grid(True)
plt.savefig('./plots/tr_times.pdf')

# Create the plot for test times
plt.figure(figsize=(10, 5))
for method, times in test_times.items():
    sizes = list(times.keys())
    time_values = list(times.values())
    plt.plot(sizes, time_values, marker='o', label=method)
plt.xlabel('Test dataset size')
plt.ylabel('Test time (s)')
plt.title('Test Times')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both")
plt.legend()
plt.grid(True)
plt.savefig('./plots/te_times.pdf')
