import os
import pickle
import matplotlib.pyplot as plt
import quapy as qp
import numpy as np

# Path to the folder containing the pickle files
folder_path = '../montecarlo_trials'
os.makedirs('./plots', exist_ok=True)

# Initialize dictionaries for training and test times
ae_by_dataset = {}
times_by_dataset = {}

# Load data from the pickle files
# for filename in os.listdir(folder_path):
for dataset in qp.datasets.UCI_MULTICLASS_DATASETS[:5]:
    with open(os.path.join(folder_path, dataset+'.pkl'), 'rb') as file:
        (test_ae, times) = pickle.load(file)
        ae_by_dataset[dataset] = test_ae
        times_by_dataset[dataset] = times

X_AXIS_LOG=True
Y_AXIS_LOG=True

# Create the plot for training times
plt.figure(figsize=(10, 5))
for dataset, times in times_by_dataset.items():
    trials = list(times.keys())
    time_values = list(times.values())
    time_values = [tr_time for (tr_time, te_time) in time_values]
    plt.plot(trials, time_values, marker='o', label=dataset)
plt.xlabel('Number of Montecarlo trials ($t$)')
plt.ylabel('Training time (s)')
plt.title('Training Times')
if X_AXIS_LOG:
    plt.xscale('log')
if Y_AXIS_LOG:
    plt.yscale('log')
plt.grid(True, which="both")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('./plots/trials_tr_times.pdf', bbox_inches='tight')

# Create the plot for test times
plt.figure(figsize=(10, 5))
for dataset, times in times_by_dataset.items():
    trials = list(times.keys())
    time_values = list(times.values())
    time_values = [te_time for (tr_time, te_time) in time_values]
    plt.plot(trials, time_values, marker='o', label=dataset)
plt.xlabel('Number of Montecarlo trials ($t$)')
plt.ylabel('Test time per sample (s)')
plt.title('Test Times')
if X_AXIS_LOG:
    plt.xscale('log')
if Y_AXIS_LOG:
    plt.yscale('log')
plt.grid(True, which="both")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('./plots/trials_te_times.pdf', bbox_inches='tight')


# Create the plot for test times
plt.figure(figsize=(10, 5))
for dataset, times in ae_by_dataset.items():
    trials = list(times.keys())
    ae_values = list(times.values())
    mae_values = [mae for (mae, mae_std) in ae_values]
    # std_values = [mae_std/np.sqrt(100) for (mae, mae_std) in ae_values]
    # plt.errorbar(trials, mae_values, yerr=std_values, marker='o', label=dataset)
    plt.plot(trials, mae_values, marker='o', label=dataset)
plt.xlabel('Number of Montecarlo trials ($t$)')
plt.ylabel('MAE')
plt.title('Performance')
if X_AXIS_LOG:
    plt.xscale('log')
# if Y_AXIS_LOG:
# plt.yscale('log')
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('./plots/trials_te_ae.pdf', bbox_inches='tight')
