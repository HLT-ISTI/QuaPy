import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

for bench in ['tweet', 'ucimulti', 'lequa/T1B']:
    print('generating face for ', bench)

    dmy = pd.read_csv(f'../results/{bench}/sensibility/DM-HD.csv', sep='\t')
    dmy = dmy.rename(columns={'nbins': 'param'})
    dmy['param name']='nbins'
    kde = pd.read_csv(f'../results/{bench}/sensibility/KDEy-ML.csv', sep='\t')
    kde = kde.rename(columns={'Bandwidth': 'param'})
    kde['param name']='bandwidth'

    df = pd.concat([dmy, kde])

    # Calcular los límites de los ejes y basados en los datos
    limits = df.groupby(['Dataset'])['MAE'].agg(['min', 'max']).reset_index()

    #cmc    DM-HD  0.06257  0.11894
    #cmc    KDEy-ML  0.09010  0.15735

    # Usar FacetGrid para crear la cuadrícula de gráficos
    g = sns.FacetGrid(df, row='Dataset', col='Method', hue='param name', margin_titles=True, height=2, aspect=2, sharex=False,  sharey=False)
    g.map(sns.lineplot, 'param', 'MAE')

    # Ajustar el layout
    g.set_axis_labels('param', 'MAE')
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    # Ajustar los límites del eje-y para cada gráfico basado en el valor máximo absoluto calculado
    for row_idx, dataset in enumerate(df['Dataset'].unique()):
        max_abs_limit = limits[limits['Dataset'] == dataset].iloc[0]
        minval, maxval = max_abs_limit['min'], max_abs_limit['max']
        amplitude = (maxval-minval)*0.05
        for col_idx, method in enumerate(df['Method'].unique()):
            ax = g.axes[row_idx, col_idx]
            ax.set_ylim(minval-amplitude, maxval+amplitude)

    datasets = df['Dataset'].unique()
    isdm = True
    for ax in g.axes.flat:
        if isdm:
            ax.set_xlim(2, 32)
        else:
            ax.set_xlim(0.01, 0.20)
        isdm = not isdm


    plt.tight_layout()
    plt.savefig(f'facet_{bench.replace('/','_')}.pdf')