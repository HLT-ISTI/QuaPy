"""
This example shows how to use Bayesian quantification (https://arxiv.org/abs/2302.09159),
which is suitable for low-data situations and when the uncertainty of the prevalence estimate is of interest.

For this, we will need to install extra dependencies:

```
$ pip install quapy[bayesian]
```

Running the script via:

```
$ python examples/bayesian_quantification.py
```

will produce a plot `bayesian_quantification.pdf`.

Due to a low sample size and the fact that classes 2 and 3 are hard to distinguish,
it is hard to estimate the proportions accurately, what is visible by looking at the posterior samples,
showing large uncertainty.
"""
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from quapy.method.aggregative import BayesianCC, ACC, PACC
from quapy.data import LabelledCollection

FIGURE_PATH = "bayesian_quantification.pdf"


@dataclass
class SimulatedData:
    """Auxiliary class to keep the training and test data sets."""
    n_classes: int
    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray


def simulate_data(rng) -> SimulatedData:
    """Generates a simulated data set with three classes."""

    # Number of examples of each class in both data sets
    n_train = [400, 400, 400]
    n_test = [40, 25, 15]

    # Mean vectors and shared covariance of P(X|Y) distributions
    mus = [np.zeros(2), np.array([1, 1.5]), np.array([1.5, 1])]
    cov = np.eye(2)
    
    # Generate the features accordingly
    X_train = np.concatenate([
        rng.multivariate_normal(mus[i], cov, size=n_train[i])
        for i in range(3)
    ])
    
    X_test = np.concatenate([
        rng.multivariate_normal(mus[i], cov, size=n_test[i])
        for i in range(3)
    ])

    Y_train = np.concatenate([[i] * n for i, n in enumerate(n_train)])
    Y_test = np.concatenate([[i] * n for i, n in enumerate(n_test)])

    return SimulatedData(
        n_classes=3,
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test,
    )


def plot_simulated_data(axs, data: SimulatedData) -> None:
    """Plots a simulated data set.
    
    Args:
        axs: a list of three `plt.Axes` objects, on which the samples will be plotted.
        data: the simulated data set.
    """
    xlim = (
        -0.3 + min(data.X_train[:, 0].min(), data.X_test[:, 0].min()),
        0.3 + max(data.X_train[:, 0].max(), data.X_test[:, 0].max())
    )
    ylim = (
        -0.3 + min(data.X_train[:, 1].min(), data.X_test[:, 1].min()),
        0.3 + max(data.X_train[:, 1].max(), data.X_test[:, 1].max())
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
        ax.scatter(data.X_train[data.Y_train == i, 0], data.X_train[data.Y_train == i, 1], c=f"C{i}", s=3, rasterized=True)

    ax = axs[1]
    ax.set_title("Test set\n(with labels)")
    for i in range(data.n_classes):
        ax.scatter(data.X_test[data.Y_test == i, 0], data.X_test[data.Y_test == i, 1], c=f"C{i}", s=3, rasterized=True)

    ax = axs[2]
    ax.set_title("Test set\n(as observed)")
    ax.scatter(data.X_test[:, 0], data.X_test[:, 1], c="C5", s=3, rasterized=True)


def get_random_forest() -> RandomForestClassifier:
    """An auxiliary factory method to generate a random forest."""
    return RandomForestClassifier(n_estimators=10, random_state=5)    


def train_and_plot_bayesian_quantification(ax: plt.Axes, training: LabelledCollection, test: np.ndarray, n_classes: int) -> None:
    """Fits Bayesian quantification and plots posterior mean as well as individual samples"""
    quantifier = BayesianCC(classifier=get_random_forest())
    quantifier.fit(training)

    # Obtain mean prediction
    mean_prediction = quantifier.quantify(test)
    x_ax = np.arange(n_classes)
    ax.plot(x_ax, mean_prediction, c="salmon", linewidth=2, linestyle=":", label="Bayesian")

    # Obtain individual samples 
    samples = quantifier.get_prevalence_samples()
    for sample in samples[::5, :]:
        ax.plot(x_ax, sample, c="salmon", alpha=0.1, linewidth=0.3, rasterized=True)


def _get_estimate(estimator_class, training: LabelledCollection, test: np.ndarray) -> None:
    """Auxiliary method for running ACC and PACC."""
    estimator = estimator_class(get_random_forest())
    estimator.fit(training)
    return estimator.quantify(test)


def train_and_plot_acc(ax: plt.Axes, training: LabelledCollection, test: np.ndarray, n_classes: int) -> None:
    estimate = _get_estimate(ACC, training, test)
    ax.plot(np.arange(n_classes), estimate, c="darkblue", linewidth=2, linestyle=":", label="ACC")


def train_and_plot_pacc(ax: plt.Axes, training: LabelledCollection, test: np.ndarray, n_classes: int) -> None:
    estimate = _get_estimate(PACC, training, test)
    ax.plot(np.arange(n_classes), estimate, c="limegreen", linewidth=2, linestyle=":", label="PACC")


def plot_true_proportions(ax: plt.Axes, test_labels: np.ndarray, n_classes: int) -> None:
    """Plots the true proportions."""
    counts = np.bincount(test_labels, minlength=n_classes)
    proportion = counts / counts.sum()
    
    x_ax = np.arange(n_classes)
    ax.plot(x_ax, proportion, c="black", linewidth=2, label="True")

    ax.set_xlabel("Class")
    ax.set_ylabel("Prevalence")
    ax.set_xticks(x_ax, x_ax + 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.1, n_classes - 0.9)
    ax.set_ylim(-0.01, 1.01)



def main() -> None:
    # --- Simulate data ---
    rng = np.random.default_rng(42)
    data = simulate_data(rng)

    # --- Plot simulated data ---
    fig, axs = plt.subplots(1, 4, figsize=(13, 3), dpi=300)
    for ax in axs:
        ax.spines[['top', 'right']].set_visible(False)
    plot_simulated_data(axs[:3], data)

    # --- Plot quantification results ---
    ax = axs[3]
    plot_true_proportions(ax, test_labels=data.Y_test, n_classes=data.n_classes)
    
    training = LabelledCollection(data.X_train, data.Y_train)
    train_and_plot_acc(ax, training=training, test=data.X_test, n_classes=data.n_classes)
    train_and_plot_pacc(ax, training=training, test=data.X_test, n_classes=data.n_classes)
    train_and_plot_bayesian_quantification(ax=ax, training=training, test=data.X_test, n_classes=data.n_classes)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH)


if __name__ == '__main__':
    main()
