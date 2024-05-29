import numpy as np

import quapy as qp
from sklearn.linear_model import LogisticRegression

from quapy.model_selection import GridSearchQ
from quapy.evaluation import evaluation_report


print('Quantifying the IFCB dataset with PACC\n')

# model selection
print('loading dataset for model selection...', end='')
train, val_gen = qp.datasets.fetch_IFCB(for_model_selection=True, single_sample_train=True)
print('[done]')
print(f'\ttraining size={len(train)}, features={train.X.shape[1]}, classes={train.n_classes}')
print(f'\tvalidation samples={val_gen.total()}')

print('model selection starts')
quantifier = qp.method.aggregative.PACC(LogisticRegression())

mod_sel = GridSearchQ(
    quantifier,
    param_grid={
        'classifier__C': np.logspace(-3,3,7),
        'classifier__class_weight': [None, 'balanced']
    },
    protocol=val_gen,
    refit=False,
    n_jobs=-1,
    verbose=True,
    raise_errors=True
).fit(train)

print(f'model selection chose hyperparameters: {mod_sel.best_params_}')
quantifier = mod_sel.best_model_

print('loading dataset for test...', end='')
train, test_gen = qp.datasets.fetch_IFCB(for_model_selection=False, single_sample_train=True)
print('[done]')
print(f'\ttraining size={len(train)}, features={train.X.shape[1]}, classes={train.n_classes}')
print(f'\ttest samples={test_gen.total()}')

print('training on the whole dataset before test')
quantifier.fit(train)

print('testing...')
report = evaluation_report(quantifier, protocol=test_gen, error_metrics=['mae'], verbose=True)
print(report.mean())
