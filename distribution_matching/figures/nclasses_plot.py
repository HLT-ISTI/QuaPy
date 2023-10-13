import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
This script generates plots of sensibility for the number of classes
Plots results for MAE, MRAE, and KLD
The hyperparameters were set as: 
    quantifier.set_params(classifier__C=0.01, classifier__class_weight='balanced', bandwidth=0.2)
"""

methods = ['DM', 'KDEy-ML', 'EMQ']
optim = 'mae'
dfs = [pd.read_csv(f'../results/lequa/nclasses/{optim}/{method}.csv', sep='\t') for method in methods]
df = pd.concat(dfs)


for err in ['MAE', 'MRAE']:
    piv = df.pivot_table(index='nClasses', columns='Method', values=err)
    g = sns.lineplot(data=piv, markers=True, dashes=False)
    g.set(xlim=(1, 28))
    g.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    g.set_ylabel(err)
    g.set_xticks(np.linspace(1, 28, 28))
    plt.xticks(rotation=90)
    plt.grid()
    plt.savefig(f'./nclasses_{err}.pdf', bbox_inches='tight')
    plt.clf()