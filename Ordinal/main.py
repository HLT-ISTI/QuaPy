from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import quapy as qp
import numpy as np

from Ordinal.model import OrderedLogisticRegression, StackedClassifier, RegressionQuantification, RegressorClassifier, \
    LogisticAT
from quapy.method.aggregative import PACC, CC, EMQ, PCC, ACC, SLD, HDy
from quapy.data import LabelledCollection
from os.path import join
import os
from utils import load_samples, load_samples_pkl
from evaluation import nmd, mnmd
from time import time
import pickle
from tqdm import tqdm
import mord


#TODO:
# Ordinal LR, LAD -> balance sample_weight
# use BERT to extract features
# other domains? Kitchen, Electronics...
# try with the inverse of the distance
# add drift='all'


def load_test_samples():
    ids = np.load(join(datapath, domain, protocol, f'{drift}drift.test.id.npy'))
    ids = set(ids)
    pklpath = join(datapath, domain, protocol, 'test_samples')
    for sample in tqdm(load_samples_pkl(pklpath, filter=ids), total=len(ids)):
        yield sample.instances, sample.prevalence()


def load_dev_samples():
    ids = np.load(join(datapath, domain, protocol, f'{drift}drift.dev.id.npy'))
    ids = set(ids)
    pklpath = join(datapath, domain, protocol, 'dev_samples')
    for sample in tqdm(load_samples_pkl(pklpath, filter=ids), total=len(ids)):
        yield sample.instances, sample.prevalence()


def quantifiers():
    params_LR = {'C': np.logspace(-3,3,7), 'class_weight': [None, 'balanced']}
    # params_OLR = {'alpha':np.logspace(-3, 3, 7), 'class_weight': [None, 'balanced']}
    params_OLR = {'alpha': np.logspace(-3, 3, 7), 'class_weight': [None, 'balanced']}
    params_SVR = {'C': np.logspace(-3,3,7), 'class_weight': [None, 'balanced']}
    # params_SVR = {'C': np.logspace(0, 1, 2)}

    # baselines
    yield 'CC(LR)', CC(LogisticRegression()), params_LR
    yield 'PCC(LR)', PCC(LogisticRegression()), params_LR
    yield 'ACC(LR)', ACC(LogisticRegression()), params_LR
    yield 'PACC(LR)', PACC(LogisticRegression()), params_LR
    #yield 'HDy(LR)', HDy(LogisticRegression()), params_LR
    yield 'SLD(LR)', EMQ(LogisticRegression()), params_LR

    # with order-aware classifiers
    # threshold-based ordinal regression (see https://pythonhosted.org/mord/)
    yield 'CC(OLR-AT)', CC(LogisticAT()), params_OLR
    yield 'PCC(OLR-AT)', PCC(LogisticAT()), params_OLR
    yield 'ACC(OLR-AT)', ACC(LogisticAT()), params_OLR
    yield 'PACC(OLR-AT)', PACC(LogisticAT()), params_OLR
    #yield 'HDy(OLR-AT)', HDy(mord.LogisticAT()), params_OLR
    yield 'SLD(OLR-AT)', EMQ(LogisticAT()), params_OLR
    # other options include mord.LogisticIT(alpha=1.), mord.LogisticSE(alpha=1.)

    # regression-based ordinal regression (see https://pythonhosted.org/mord/) 
    # I am using my implementation, which caters for predict_proba (linear distance to the two closest classes, 0 in the rest)
    # the other implementation has OrdinalRidge(alpha=1.0) and LAD(C=1.0) with my wrapper classes for having the nclasses_; those do
    # not implement predict_proba nor decision_score
    yield 'CC(SVR)', CC(RegressorClassifier()), params_SVR
    yield 'CC-bal(SVR)', CC(RegressorClassifier()), params_SVR
    # yield 'PCC(SVR)', PCC(RegressorClassifier()), params_SVR
    # yield 'PCC-cal(SVR)', PCC(RegressorClassifier()), params_SVR
    # yield 'ACC(SVR)', ACC(RegressorClassifier()), params_SVR
    # yield 'PACC(SVR)', PACC(RegressorClassifier()), params_SVR
    #yield 'HDy(SVR)', HDy(RegressorClassifier()), params_SVR
    # yield 'SLD(SVR)', EMQ(RegressorClassifier()), params_SVR


def run_experiment(params):
    qname, q, param_grid, drift = params
    resultfile = join(resultpath, f'{qname}.{drift}.csv')
    if os.path.exists(resultfile):
        print(f'result file {resultfile} already exists: continue')
        return None

    print(f'fitting {qname} for {drift}-drift')

    q = qp.model_selection.GridSearchQ(
        q,
        param_grid,
        sample_size=1000,
        protocol='gen',
        error=mnmd,
        val_split=load_dev_samples,
        n_jobs=-1,
        refit=False,
        verbose=True).fit(train)

    hyperparams = f'{qname}\t{drift}\t{q.best_params_}'

    print('[done]')

    report = qp.evaluation.gen_prevalence_report(q, gen_fn=load_test_samples, error_metrics=[nmd])
    mean_nmd = report['nmd'].mean()
    std_nmd = report['nmd'].std()
    print(f'{qname}: {mean_nmd:.4f} +-{std_nmd:.4f}')
    report.to_csv(resultfile, index=False)

    print('[learning regressor-based adjustment]')
    q = RegressionQuantification(q.best_model(), val_samples_generator=load_dev_samples)
    q.fit(None)

    report = qp.evaluation.gen_prevalence_report(q, gen_fn=load_test_samples, error_metrics=[nmd])
    mean_nmd = report['nmd'].mean()
    std_nmd = report['nmd'].std()
    print(f'[{qname} regression-correction] {mean_nmd:.4f} +-{std_nmd:.4f}')
    resultfile = join(resultpath, f'{qname}.{drift}.reg.csv')
    report.to_csv(resultfile, index=False)

    return hyperparams


if __name__ == '__main__':
    domain = 'Books-tfidf'
    datapath = './data'
    protocol = 'app'
    resultpath = join('./results', domain, protocol)
    os.makedirs(resultpath, exist_ok=True)

    train = pickle.load(open(join(datapath, domain, 'training_data.pkl'), 'rb'))

    with open(join(resultpath, 'hyper.txt'), 'at') as foo:
        for drift in ['low', 'mid', 'high', 'all']:
            params = [(*qs, drift) for qs in quantifiers()]
            hypers = qp.util.parallel(run_experiment, params, n_jobs=-2)
            for h in hypers:
                if h is not None:
                    foo.write(h)
                    foo.write('\n')





