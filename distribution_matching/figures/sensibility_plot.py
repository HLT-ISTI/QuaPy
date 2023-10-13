import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
This script generates plots of sensibility for the hyperparameter "bandwidth".
Plots results for MAE, MRAE, and KLD
The rest of hyperparameters were set to their default values
"""

df_tweet = pd.read_csv('../results/tweet/sensibility/KDEy-ML.csv', sep='\t')
df_lequa = pd.read_csv('../results/lequa/sensibility/KDEy-ML.csv', sep='\t')
df = pd.concat([df_tweet, df_lequa])

for err in ['MAE', 'MRAE', 'KLD']:
    piv = df.pivot_table(index='Bandwidth', columns='Dataset', values=err)
    g = sns.lineplot(data=piv, markers=True, dashes=False)
    g.set(xlim=(0.01, 0.2))
    g.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    g.set_ylabel(err)
    g.set_xticks(np.linspace(0.01, 0.2, 20))
    plt.xticks(rotation=90)
    plt.grid()
    plt.savefig(f'./sensibility_{err}.pdf', bbox_inches='tight')
    plt.clf()