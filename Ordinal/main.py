from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import quapy as qp
import numpy as np

from Ordinal.model import OrderedLogisticRegression, StackedClassifier, RegressionQuantification, RegressorClassifier
from quapy.method.aggregative import PACC, CC, EMQ, PCC, ACC
from quapy.data import LabelledCollection
from os.path import join
from utils import load_samples, load_samples_pkl
from evaluation import nmd, mnmd
from time import time
import pickle
from tqdm import tqdm

domain = 'Books-tfidf'
datapath = './data'
protocol = 'app'
drift = 'high'

train = pickle.load(open(join(datapath, domain, 'training_data.pkl'), 'rb'))


def load_test_samples():
    ids = np.load(join(datapath, domain, protocol, f'{drift}drift.test.id.npy'))
    ids = set(ids)
    for sample in tqdm(load_samples_pkl(join(datapath, domain, protocol, 'test_samples'), filter=ids), total=len(ids)):
        yield sample.instances, sample.prevalence()


def load_dev_samples():
    ids = np.load(join(datapath, domain, protocol, f'{drift}drift.dev.id.npy'))
    ids = set(ids)
    for sample in tqdm(load_samples_pkl(join(datapath, domain, protocol, 'dev_samples'), filter=ids), total=len(ids)):
        yield sample.instances, sample.prevalence()


print('fitting the quantifier')

# q = EMQ(LogisticRegression(class_weight='balanced'))
# q = PACC(LogisticRegression(class_weight='balanced'))
q = PACC(OrderedLogisticRegression())
# q = PACC(StackedClassifier(LogisticRegression(class_weight='balanced')))
# q = RegressionQuantification(PCC(LogisticRegression(class_weight='balanced')), val_samples_generator=load_dev_samples)
# q = ACC(RegressorClassifier())

param_grid = {'C': np.logspace(-3,3,7), 'class_weight': [None, 'balanced']}
# param_grid = {'C': np.logspace(-3,3,14)}
# param_grid = {'alpha':np.logspace(-8, 6, 15)}

# q = qp.model_selection.GridSearchQ(
#     q,
#     param_grid,
#     1000,
#     'gen',
#     error=mnmd,
#     val_split=load_dev_samples,
#     n_jobs=-1,
#     refit=False,
#     verbose=True)

q.fit(train)

# q = RegressionQuantification(q, val_samples_generator=load_dev_samples)
# q.fit(None)

print('[done]')

report = qp.evaluation.gen_prevalence_report(q, gen_fn=load_test_samples, error_metrics=[nmd])
mean_nmd = report['nmd'].mean()
std_nmd = report['nmd'].std()
print(f'{mean_nmd:.4f} +-{std_nmd:.4f}')

q = RegressionQuantification(q, val_samples_generator=load_dev_samples)
q.fit(None)

report = qp.evaluation.gen_prevalence_report(q, gen_fn=load_test_samples, error_metrics=[nmd])
mean_nmd = report['nmd'].mean()
std_nmd = report['nmd'].std()
print(f'[regression-correction] {mean_nmd:.4f} +-{std_nmd:.4f}')

# drift='high'
# report = qp.evaluation.gen_prevalence_report(q, gen_fn=load_test_samples, error_metrics=[nmd])
# mean_nmd = report['nmd'].mean()
# std_nmd = report['nmd'].std()
# print(f'{mean_nmd:.4f} +-{std_nmd:.4f}')



