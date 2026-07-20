import numpy as np
from sklearn.datasets import make_classification

from quapy.data import LabelledCollection
from quapy.data.base import Dataset


def make_labelled_collection(
    n_samples=200,
    n_features=12,
    n_classes=2,
    class_sep=1.5,
    random_state=0,
):
    n_informative = min(n_features, max(4, n_classes * 2))
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=class_sep,
        random_state=random_state,
    )
    classes = np.arange(n_classes)
    return LabelledCollection(X, y, classes=classes)


def make_dataset(
    n_train=150,
    n_test=80,
    n_features=12,
    n_classes=2,
    class_sep=1.5,
    random_state=0,
    name='synthetic',
):
    data = make_labelled_collection(
        n_samples=n_train + n_test,
        n_features=n_features,
        n_classes=n_classes,
        class_sep=class_sep,
        random_state=random_state,
    )
    training, test = data.split_stratified(train_prop=n_train / (n_train + n_test), random_state=random_state)
    return Dataset(training, test, name=name)
