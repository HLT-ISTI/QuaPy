from sklearn.linear_model import LogisticRegression
import quapy as qp
from method.aggregative import *

datasets = qp.datasets.UCI_MULTICLASS_DATASETS[1]
data = qp.datasets.fetch_UCIMulticlassDataset(datasets)
train, test = data.train_test

Xtr, ytr = train.Xy
Xte = test.X

quant = EMQ(LogisticRegression(), calib='bcts')
quant.fit(Xtr, ytr)
prev = quant.predict(Xte)

print(prev)
post = quant.predict_proba(Xte)
print(post)
post = quant.classify(Xte)
print(post)

# AggregativeMedianEstimator()


# test CC, prevent from doing 5FCV for nothing
# test PACC o PCC with LinearSVC; removing "adapt_if_necessary" form _check_classifier