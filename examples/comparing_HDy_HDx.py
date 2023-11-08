from sklearn.linear_model import LogisticRegression
from time import time
import pandas as pd
from tqdm import tqdm

import quapy as qp
from quapy.protocol import APP
from quapy.method.aggregative import HDy
from quapy.method.non_aggregative import HDx


"""
This example is meant to experimentally compare HDy and HDx. 
The implementations of these methods adhere to the original design of the methods; in particular, this means that
the number of bins is not an hyperparameter, but is something that the method explores internally (returning the
median of the estimates as the final prevalence prediction), and the prevalence is not searched through any 
numerical optimization procedure, but simply as a linear search between 0 and 1 steppy by 0.01.
See <https://www.sciencedirect.com/science/article/pii/S0020025512004069>`_ for further details   
"""

qp.environ['SAMPLE_SIZE']=100


df = pd.DataFrame(columns=('method', 'dataset', 'MAE', 'MRAE', 'tr-time', 'te-time'))


for dataset_name in tqdm(qp.datasets.UCI_DATASETS, total=len(qp.datasets.UCI_DATASETS)):
    if dataset_name in ['acute.a', 'acute.b', 'balance.2', 'iris.1']: continue

    collection = qp.datasets.fetch_UCILabelledCollection(dataset_name, verbose=False)
    train, test = collection.split_stratified()

    # HDy............................................
    tinit = time()
    hdy = HDy(LogisticRegression()).fit(train)
    t_hdy_train = time()-tinit

    tinit = time()
    hdy_report = qp.evaluation.evaluation_report(hdy, APP(test), error_metrics=['mae', 'mrae']).mean()
    t_hdy_test = time() - tinit
    df.loc[len(df)] = ['HDy', dataset_name, hdy_report['mae'], hdy_report['mrae'], t_hdy_train, t_hdy_test]

    # HDx............................................
    tinit = time()
    hdx = HDx().fit(train)
    t_hdx_train = time() - tinit

    tinit = time()
    hdx_report = qp.evaluation.evaluation_report(hdx, APP(test), error_metrics=['mae', 'mrae']).mean()
    t_hdx_test = time() - tinit
    df.loc[len(df)] = ['HDx', dataset_name, hdx_report['mae'], hdx_report['mrae'], t_hdx_train, t_hdx_test]

# evaluation reports

print('\n'*3)
print('='*80)
print('Comparison in terms of performance')
print('='*80)
pv = df.pivot_table(index='dataset', columns='method', values=['MAE', 'MRAE'])
print(pv)
print('\nAveraged values:')
print(pv.mean())

print('\n'*3)
print('='*80)
print('Comparison in terms of efficiency')
print('='*80)
pv = df.pivot_table(index='dataset', columns='method', values=['tr-time', 'te-time'])
print(pv)
print('\nAveraged values:')
print(pv.mean())



