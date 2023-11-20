import pickle
import os

import quapy as qp
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP


toprint = []
for dataset in qp.datasets.UCI_MULTICLASS_DATASETS:

    data = qp.datasets.fetch_UCIMulticlassDataset(dataset)

    # model selection
    train, test = data.train_test
    toprint.append(f'{dataset}\t{len(train)}\t{len(test)}\t{data.n_classes}')

print()
for pr in toprint:
    print(pr)
