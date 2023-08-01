import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import sys
import pandas as pd

import quapy as qp
from quapy.method.aggregative import EMQ, DistributionMatching, PACC, HDy, OneVsAllAggregative, ACC
from method_kdey import KDEy
from method_dirichlety import DIRy
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP


if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = qp.datasets.LEQUA2022_SAMPLE_SIZE['T1B']
    qp.environ['N_JOBS'] = -1
    optim = 'mrae'
    result_dir = f'results_lequa_{optim}'

    os.makedirs(result_dir, exist_ok=True)

    hyper_LR = {
        'classifier__C': np.logspace(-3,3,7),
        'classifier__class_weight': ['balanced', None]
    } 

    for method in ['DIR']:#'HDy-OvA', 'SLD', 'ACC-tv', 'PACC-tv']: #['DM', 'DIR']: #'KDEy-MLE', 'KDE-DM', 'DM', 'DIR']:
        
        print('Init method', method)

        result_path = f'{result_dir}/{method}'
        
        if os.path.exists(result_path+'.csv'):
            print(f'file {result_path}.csv already exist; skipping')
            continue

        with open(result_path+'.csv', 'wt') as csv:
            csv.write(f'Method\tDataset\tMAE\tMRAE\tKLD\n')    

            dataset = 'T1B'
            train, val_gen, test_gen = qp.datasets.fetch_lequa2022(dataset)
            print(f'init {dataset} #instances: {len(train)}')
            if method == 'KDEy-MLE':
                method_params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
                param_grid = {**method_params, **hyper_LR}
                quantifier = KDEy(LogisticRegression(), target='max_likelihood', val_split=10)
            elif method in ['KDE-DM']:
                method_params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
                param_grid = {**method_params, **hyper_LR}
                quantifier = KDEy(LogisticRegression(), target='min_divergence', divergence='l2', montecarlo_trials=5000, val_split=10)
            elif method == 'DIR':
                param_grid = hyper_LR
                quantifier = DIRy(LogisticRegression())
            elif method == 'SLD':
                param_grid = hyper_LR
                quantifier = EMQ(LogisticRegression())
            elif method == 'PACC-tv':
                param_grid = hyper_LR
                quantifier = PACC(LogisticRegression())
            elif method == 'ACC-tv':
                param_grid = hyper_LR
                quantifier = ACC(LogisticRegression())
            elif method == 'HDy-OvA':
                param_grid = {'binary_quantifier__' + key: val for key, val in hyper_LR.items()}
                quantifier = OneVsAllAggregative(HDy(LogisticRegression()))
            elif method == 'DM':
                method_params = {
                    'nbins': [4,8,16,32],
                    'val_split': [10, 0.4],
                    'divergence': ['HD', 'topsoe', 'l2']
                }
                param_grid = {**method_params, **hyper_LR}
                quantifier = DistributionMatching(LogisticRegression())
            else:
                raise NotImplementedError('unknown method', method)

            if param_grid is not None:
                modsel = GridSearchQ(quantifier, param_grid, protocol=val_gen, refit=False, n_jobs=-1, verbose=1, error=optim)

                modsel.fit(train)
                print(f'best params {modsel.best_params_}')
                print(f'best score {modsel.best_score_}')
                pickle.dump(
                    (modsel.best_params_, modsel.best_score_,), 
                    open(f'{result_path}.hyper.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

                quantifier = modsel.best_model()
            else:
                print('debug mode... skipping model selection')
                quantifier.fit(train)

            report = qp.evaluation.evaluation_report(quantifier, protocol=test_gen, error_metrics=['mae', 'mrae', 'kld'], verbose=True)
            means = report.mean()
            report.to_csv(result_path+'.dataframe')
            csv.write(f'{method}\tLeQua-T1B\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{means["kld"]:.5f}\n')
            csv.flush()

    df = pd.read_csv(result_path+'.csv', sep='\t')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["MAE", "MRAE"])
    print(pv)
