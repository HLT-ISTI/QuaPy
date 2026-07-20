import unittest

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

import numpy as np

import quapy as qp


@unittest.skipUnless(HAS_MATPLOTLIB and qp.plot is not None, 'matplotlib is not available')
class TestPlot(unittest.TestCase):

    def test_plot_simplex_smoke(self):
        rng = np.random.default_rng(0)
        true_prev = np.array([0.2, 0.3, 0.5])
        cloud = rng.dirichlet(alpha=30 * true_prev, size=50)

        fig, ax = plt.subplots(figsize=(5, 5))
        fig, ax = qp.plot.plot_simplex(
            point_layers=[
                {'points': cloud, 'label': 'cloud', 'style': {'s': 8, 'alpha': 0.2}},
                {'points': true_prev, 'label': 'target', 'style': {'s': 50, 'color': 'black'}},
            ],
            region_layers=[
                {'fn': lambda p: p[:, 2] >= 0.4, 'label': 'high class-3', 'color': 'green', 'alpha': 0.2},
            ],
            density_function=lambda p: np.exp(-25 * np.sum((p - true_prev) ** 2, axis=1)),
            class_names=['A', 'B', 'C'],
            ax=ax,
        )

        self.assertIs(fig, ax.figure)
        self.assertGreaterEqual(len(ax.collections), 2)
        plt.close(fig)
