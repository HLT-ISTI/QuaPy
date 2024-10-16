from sklearn.linear_model import LogisticRegression
from statsmodels.sandbox.distributions.genpareto import quant

import quapy as qp
from quapy.protocol import UPP
from quapy.method.aggregative import PACC, DMy, EMQ, KDEyML
from quapy.method.meta import SCMQ

qp.environ["SAMPLE_SIZE"]=100

def train_and_test_model(quantifier, train, test):
    quantifier.fit(train)
    report = qp.evaluation.evaluation_report(quantifier, UPP(test), error_metrics=['mae', 'mrae'])
    print(quantifier.__class__.__name__)
    print(report.mean(numeric_only=True))


quantifiers = [
    PACC(),
    DMy(),
    EMQ(),
    KDEyML()
]

classifier = LogisticRegression()

dataset_name = qp.datasets.UCI_MULTICLASS_DATASETS[0]
data = qp.datasets.fetch_UCIMulticlassDataset(dataset_name)
train, test = data.train_test

scmq = SCMQ(classifier, quantifiers)

train_and_test_model(scmq, train, test)

for quantifier in quantifiers:
    train_and_test_model(quantifier, train, test)