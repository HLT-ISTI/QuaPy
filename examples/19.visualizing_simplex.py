import numpy as np
import quapy as qp
from quapy.method.confidence import ConfidenceIntervals


"""
A minimal example showing how to visualise ternary prevalences on the simplex.
The plot combines a cloud of posterior triplets, a few reference prevalences, a
confidence ellipse induced by the cloud, and a smooth density centred around the
true prevalence.
"""

rng = np.random.default_rng(0)
true_prev = np.array([0.20, 0.35, 0.45])
train_prev = np.array([0.50, 0.30, 0.20])
pred_prev = np.array([0.18, 0.39, 0.43])
posterior_cloud = rng.dirichlet(alpha=45 * true_prev, size=250)

point_layers = [
    {
        'points': posterior_cloud,
        'label': 'posterior cloud',
        'style': {'s': 12, 'alpha': 0.25, 'color': 'steelblue', 'edgecolors': 'none'},
    },
    {
        'points': true_prev,
        'label': 'true prevalence',
        'style': {'s': 70, 'color': 'black'},
    },
    {
        'points': pred_prev,
        'label': 'predicted prevalence',
        'style': {'s': 70, 'color': 'crimson'},
    },
    {
        'points': train_prev,
        'label': 'training prevalence',
        'style': {'s': 70, 'color': 'darkorange'},
    },
]

confidence_region = ConfidenceIntervals(posterior_cloud, confidence_level=0.95, bonferroni_correction=True)

region_layers = [
    {
        'fn': lambda p: float(p in confidence_region),
        'label': '95% confidence intervals',
        'color': 'seagreen',
        'alpha': 0.15,
    }
]

density = lambda p: np.exp(-45 * np.sum((p - true_prev) ** 2, axis=1))

qp.plot.plot_simplex(
    point_layers=point_layers,
    region_layers=region_layers,
    density_function=density,
    density_color='royalblue',
    class_names=['class A', 'class B', 'class C'],
    title='Ternary prevalence visualisation',
    legend_ncol=3,
    figsize=(7.2, 5.8),
    class_name_fontsize=9,
    title_fontsize=10,
    legend_fontsize=8,
    savepath='./plots/simplex_visualization.png',
)
