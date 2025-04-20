import quapy as qp
from method.aggregative import *

datasets = qp.datasets.UCI_MULTICLASS_DATASETS[1]
data = qp.datasets.fetch_UCIMulticlassDataset(datasets)
train, test = data.train_test

quant = EMQ()
quant.fit(*train.Xy)
prev = quant.predict(test.X)

print(prev)


# test CC, prevent from doing 5FCV for nothing
# test PACC o PCC with LinearSVC; removing "adapt_if_necessary" form _check_classifier