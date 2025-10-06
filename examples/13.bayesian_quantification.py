"""
.. author:: Paweł Czyż

This example shows how to use Bayesian quantification (https://arxiv.org/abs/2302.09159),
which is suitable for low-data situations and when the uncertainty of the prevalence estimate is of interest.

For this, we will need to install extra dependencies:

```
$ pip install quapy[bayesian]
```

Running the script via:

```
$ python examples/13.bayesian_quantification.py
```

will produce a plot `bayesian_quantification.pdf`.

Due to a low sample size and the fact that classes 2 and 3 are hard to distinguish,
it is hard to estimate the proportions accurately, what is visible by looking at the posterior samples,
showing large uncertainty.
"""

import numpy as np
import matplotlib.pyplot as plt
import quapy as qp

from sklearn.ensemble import RandomForestClassifier

from quapy.method.aggregative import BayesianCC, ACC, PACC
from quapy.data import LabelledCollection, Dataset


FIGURE_PATH = "bayesian_quantification.pdf"


def simulate_data(rng) -> Dataset:
    """Generates a simulated data set with three classes."""

    # Number of examples of each class in both data sets
    n_train = [400, 400, 400]
    n_test = [40, 25, 15]

    # Mean vectors and shared covariance of P(X|Y) distributions
    mus = [np.zeros(2), np.array([1, 1.5]), np.array([1.5, 1])]
    cov = np.eye(2)

    def gen_Xy(centers, sizes):
        X = np.concatenate([rng.multivariate_normal(mu_i, cov, size_i) for mu_i, size_i in zip(centers, sizes)])
        y = np.concatenate([[i] * n for i, n in enumerate(sizes)])
        return X, y

    # Generate the features accordingly
    train = LabelledCollection(*gen_Xy(centers=mus, sizes=n_train))
    test  = LabelledCollection(*gen_Xy(centers=mus, sizes=n_test))
    
    return Dataset(training=train, test=test)


def plot_simulated_data(axs, data: Dataset) -> None:
    """Plots a simulated data set.

    :param axs: a list of three `plt.Axes` objects, on which the samples will be plotted.
    :param data: the simulated data set.
    """
    train, test = data.train_test
    xlim = (
        -0.3 + min(train.X[:, 0].min(), test.X[:, 0].min()),
        0.3 + max(train.X[:, 0].max(), test.X[:, 0].max())
    )
    ylim = (
        -0.3 + min(train.X[:, 1].min(), test.X[:, 1].min()),
        0.3 + max(train.X[:, 1].max(), test.X[:, 1].max())
    )

    for ax in axs:
        ax.set_xlabel("$X_1$")
        ax.set_ylabel("$X_2$")
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axs[0]
    ax.set_title("Training set")
    for i in range(data.n_classes):
        ax.scatter(train.X[train.y == i, 0], train.X[train.y == i, 1], c=f"C{i}", s=3, rasterized=True)

    ax = axs[1]
    ax.set_title("Test set\n(with labels)")
    for i in range(data.n_classes):
        ax.scatter(test.X[test.y == i, 0], test.X[test.y == i, 1], c=f"C{i}", s=3, rasterized=True)

    ax = axs[2]
    ax.set_title("Test set\n(as observed)")
    ax.scatter(test.X[:, 0], test.X[:, 1], c="C5", s=3, rasterized=True)


def plot_true_proportions(ax: plt.Axes, test_prevalence: np.ndarray) -> None:
    """Plots the true proportions."""
    n_classes = len(test_prevalence)
    x_ax = np.arange(n_classes)
    ax.plot(x_ax, test_prevalence, c="black", linewidth=2, label="True")

    ax.set_xlabel("Class")
    ax.set_ylabel("Prevalence")
    ax.set_xticks(x_ax, x_ax + 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.1, n_classes - 0.9)
    ax.set_ylim(-0.01, 1.01)


def get_random_forest() -> RandomForestClassifier:
    """An auxiliary factory method to generate a random forest."""
    return RandomForestClassifier(n_estimators=10, random_state=5)    


def _get_estimate(estimator_class, training: LabelledCollection, test: np.ndarray) -> None:
    """Auxiliary method for running ACC and PACC."""
    estimator = estimator_class(get_random_forest())
    estimator.fit(training)
    return estimator.quantify(test)


def train_and_plot_bayesian_quantification(ax: plt.Axes, training: LabelledCollection, test: LabelledCollection) -> None:
    """Fits Bayesian quantification and plots posterior mean as well as individual samples"""
    print('training model Bayesian CC...', end='')
    quantifier = BayesianCC(classifier=get_random_forest())
    quantifier.fit(training)

    # Obtain mean prediction
    mean_prediction = quantifier.quantify(test.X)
    mae = qp.error.mae(test.prevalence(), mean_prediction)
    x_ax = np.arange(training.n_classes)
    ax.plot(x_ax, mean_prediction, c="salmon", linewidth=2, linestyle=":", label="Bayesian")

    # Obtain individual samples 
    samples = quantifier.get_prevalence_samples()
    for sample in samples[::5, :]:
        ax.plot(x_ax, sample, c="salmon", alpha=0.1, linewidth=0.3, rasterized=True)
    print(f'MAE={mae:.4f} [done]')


def train_and_plot_acc(ax: plt.Axes, training: LabelledCollection, test: LabelledCollection) -> None:
    print('training model ACC...', end='')
    estimate = _get_estimate(ACC, training, test.X)
    mae = qp.error.mae(test.prevalence(), estimate)
    ax.plot(np.arange(training.n_classes), estimate, c="darkblue", linewidth=2, linestyle=":", label="ACC")
    print(f'MAE={mae:.4f} [done]')


def train_and_plot_pacc(ax: plt.Axes, training: LabelledCollection, test: LabelledCollection) -> None:
    print('training model PACC...', end='')
    estimate = _get_estimate(PACC, training, test.X)
    mae = qp.error.mae(test.prevalence(), estimate)
    ax.plot(np.arange(training.n_classes), estimate, c="limegreen", linewidth=2, linestyle=":", label="PACC")
    print(f'MAE={mae:.4f} [done]')


def main() -> None:
    # --- Simulate data ---
    print('generating simulated data')
    rng = np.random.default_rng(42)
    data = simulate_data(rng)
    training, test = data.train_test

    # --- Plot simulated data ---
    fig, axs = plt.subplots(1, 4, figsize=(13, 3), dpi=300)
    for ax in axs:
        ax.spines[['top', 'right']].set_visible(False)
    plot_simulated_data(axs[:3], data)

    # --- Plot quantification results ---
    ax = axs[3]
    plot_true_proportions(ax, test_prevalence=test.prevalence())

    train_and_plot_acc(ax, training=training, test=test)
    train_and_plot_pacc(ax, training=training, test=test)
    train_and_plot_bayesian_quantification(ax=ax, training=training, test=test)
    print('[done]')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    print(f'saving plot in path {FIGURE_PATH}...', end='')
    fig.tight_layout()
    fig.savefig(FIGURE_PATH)
    print('[done]')


if __name__ == '__main__':
    main()
