import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from distribution_matching.scalability_analysis import methods

# Path to the folder containing the pickle files
folder_path = '../times'
os.makedirs('./plots', exist_ok=True)

# Initialize dictionaries for training and test times
train_times = {}
test_times = {}

# Load data from the pickle files
for method, _ in methods():
    with open(os.path.join(folder_path, method + '.pkl'), 'rb') as file:
        (tr_times, te_times) = pickle.load(file)
        train_times[method] = tr_times
        test_times[method] = te_times

# Function to convert the dictionary to a DataFrame
def convert_to_dataframe(times_dict, time_type):
    data = []
    for method, times in times_dict.items():
        for size, time in times.items():
            data.append({'Method': method, 'Size': size, 'Time': time, 'Type': time_type})
    return pd.DataFrame(data)

# Convert dictionaries to DataFrames
train_df = convert_to_dataframe(train_times, 'Training')
test_df = convert_to_dataframe(test_times, 'Test')

# Create the plot for training times
plt.figure(figsize=(5, 5))
sns.lineplot(data=train_df, x='Size', y='Time', hue='Method', style='Method', markers=True, dashes=False)
plt.xlabel('Training dataset size')
plt.ylabel('Training time (s)')
plt.title('Training Times')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both")
plt.legend().remove()
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('./plots/tr_times.pdf', bbox_inches='tight')
plt.close()

# Create the plot for test times
plt.figure(figsize=(5, 5))
sns.lineplot(data=test_df, x='Size', y='Time', hue='Method', style='Method', markers=True, dashes=False)
plt.xlabel('Test dataset size')
plt.ylabel('Test time (s)')
plt.title('Test Times')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('./plots/te_times.pdf', bbox_inches='tight')
plt.close()
