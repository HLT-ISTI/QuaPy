import numpy as np
import quapy as qp
from quapy.data.datasets import fetch_UCIMulticlassDataset
from quapy.protocol import APP, NPP, UPP, DirichletProtocol


"""
Ternary plots showcasing different sampling protocols.
"""

rng = np.random.default_rng(0)

train, test = fetch_UCIMulticlassDataset(dataset_name='academic-success').train_test

train_prev = {
    'points': train.prevalence(),
    'label': 'training prevalence',
    'style': {'s': 70, 'color': 'darkorange'},
}

def protocols():
    yield 'app-grid', 'Artificial Prevalence Protocol (grid)', APP(test, n_prevalences=21, repeats=1, sample_size=100)
    yield 'app-kraemer', 'Artificial Prevalence Protocol (Kraemer)', UPP(test, repeats=5000, sample_size=500)
    yield 'npp', 'Natural Prevalence Protocol', NPP(test, repeats=1000, sample_size=100)
    yield 'dirichlet', 'Dirichlet(alpha=0.2)', DirichletProtocol(test, alpha=0.2, repeats=5000, sample_size=100)

for file_name, prot_name, protocol in protocols():
    app_points = {
        'points': [prev for _, prev in protocol()],
        'label': prot_name,
        'style': {'s': 15, 'alpha': 0.5, 'color': 'steelblue', 'edgecolors': 'none'},
    }

    point_layers = [        
        app_points,
        train_prev,
    ]

    dispersion = 0.1

    qp.plot.plot_simplex(
        point_layers=point_layers,
        class_names=['class A', 'class B', 'class C'],
        #title='Ternary prevalence visualisation',
        legend_ncol=3,
        figsize=(7.2, 5.8),
        class_name_fontsize=9,
        title_fontsize=10,
        legend_fontsize=8,
        savepath=f'./plots/{file_name}.png',
    )
