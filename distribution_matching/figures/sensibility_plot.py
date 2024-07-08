import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
This script generates plots of sensibility for the hyperparameter "bandwidth".
Plots results for MAE, MRAE, and KLD
The rest of hyperparameters were set to their default values
"""



log_mrae = True

for method, param, xlim, xticks in [
    ('KDEy-ML', 'Bandwidth', (0.01, 0.2), np.linspace(0.01, 0.2, 20)),
    ('DM-HD', 'nbins', (2,32), list(range(2,10)) + list(range(10,34,2)))
]:

    for dataset in ['tweet', 'lequa', 'uciml']:

        if dataset == 'tweet':
            df = pd.read_csv(f'../results/tweet/sensibility/{method}.csv', sep='\t')
            ylim = (0.03, 0.21)
        elif dataset == 'lequa':
            df = pd.read_csv(f'../results/lequa/T1B/sensibility/{method}.csv', sep='\t')
            ylim = (0.0125, 0.03)
        elif dataset == 'uciml':
            ylim = (0, 0.4)
            df = pd.read_csv(f'../results/ucimulti/sensibility/{method}.csv', sep='\t')

        for err in ['MAE']: #, 'MRAE']:
            piv = df.pivot_table(index=param, columns='Dataset', values=err)
            g = sns.lineplot(data=piv, markers=True, dashes=False)
            g.set(xlim=xlim)
            g.legend(loc="center left", bbox_to_anchor=(1, 0.5))

            if log_mrae and err=='MRAE':
                plt.yscale('log')
                g.set_ylabel('log('+err+')')
            else:
                g.set_ylabel(err)

            #g.set_ylim(ylim)
            g.set_xticks(xticks)
            plt.xticks(rotation=90)
            plt.yscale('log')
            plt.grid()
            plt.savefig(f'./sensibility_{method}_{dataset}_{err}.pdf', bbox_inches='tight')
            plt.clf()
