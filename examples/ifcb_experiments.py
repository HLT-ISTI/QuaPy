import quapy as qp
from sklearn.linear_model import LogisticRegression
from quapy.evaluation import evaluation_report


def newLR():
    return LogisticRegression(n_jobs=-1)


quantifiers = [
    ('CC', qp.method.aggregative.CC(newLR())),
    ('ACC', qp.method.aggregative.ACC(newLR())),
    ('PCC', qp.method.aggregative.PCC(newLR())),
    ('PACC', qp.method.aggregative.PACC(newLR())),
    ('HDy', qp.method.aggregative.DMy(newLR())),
    ('EMQ', qp.method.aggregative.EMQ(newLR()))
]


for quant_name, quantifier in quantifiers:
    print("Experiment with "+quant_name)

    train, test_gen = qp.datasets.fetch_IFCB()

    quantifier.fit(train)

    report = evaluation_report(quantifier, protocol=test_gen, error_metrics=['mae'], verbose=True)
    print(report.mean())
