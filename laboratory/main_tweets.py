import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import sys
import pandas as pd

import quapy as qp
from quapy.method.aggregative import EMQ, DistributionMatching, PACC, ACC, CC, PCC, HDy, OneVsAllAggregative
from method_kdey import KDEy
from method_dirichlety import DIRy
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP

SEED=1

if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = 100
    qp.environ['N_JOBS'] = -1
    n_bags_val = 250
    n_bags_test = 1000
    result_dir = f'results_tweet_{n_bags_test}'
    optim = 'mae'

    os.makedirs(result_dir, exist_ok=True)

    hyper_LR = {
        'classifier__C': np.logspace(-4,4,9),
        'classifier__class_weight': ['balanced', None]
    } 

    for method in ['KDE-nomonte', 'KDE-monte2', 'SLD', 'KDE-kfcv']:# , 'DIR', 'DM',  'HDy-OvA', 'CC', 'ACC', 'PCC']:
        
        #if os.path.exists(result_path):
        #    print('Result already exit. Nothing to do')
        #    sys.exit(0)

        result_path = f'{result_dir}/{method}'
        if os.path.exists(result_path+'.dataframe'):
            print(f'result file {result_path} already exist; skipping')
            continue 

        with open(result_path+'.csv', 'at') as csv:
            csv.write(f'Method\tDataset\tMAE\tMRAE\tKLD\n')

            # four semeval dataset share the training, so it is useless to optimize hyperparameters four times;
            # this variable controls that the mod sel has already been done, and skip this otherwise
            semeval_trained = False

            for dataset in qp.datasets.TWITTER_SENTIMENT_DATASETS_TEST:
                print('init', dataset)
                
                with qp.util.temp_seed(SEED):

                    is_semeval = dataset.startswith('semeval')

                    if not is_semeval or not semeval_trained:

                        if method == 'KDE':
                            param_grid = {
                                'bandwidth': np.linspace(0.001, 0.2, 21), 
                                'classifier__C': np.logspace(-4,4,9),
                                'classifier__class_weight': ['balanced', None]
                            }
                            quantifier = KDEy(LogisticRegression(), target='max_likelihood')
                        elif method == 'KDE-kfcv':
                            param_grid = {
                                'bandwidth': np.linspace(0.001, 0.2, 21), 
                                'classifier__C': np.logspace(-4,4,9),
                                'classifier__class_weight': ['balanced', None]
                            }
                            quantifier = KDEy(LogisticRegression(), target='max_likelihood', val_split=10)
                        elif method in ['KDE-monte2']:
                            param_grid = {
                                'bandwidth': np.linspace(0.001, 0.2, 21),
                            }
                            quantifier = KDEy(LogisticRegression(), target='min_divergence')
                        elif method in ['KDE-nomonte']:
                            param_grid = {
                                'bandwidth': np.linspace(0.001, 0.2, 21),
                            }
                            quantifier = KDEy(LogisticRegression(), target='max_likelihood')
                        elif method == 'DIR':
                            param_grid = hyper_LR
                            quantifier = DIRy(LogisticRegression())
                        elif method == 'SLD':
                            param_grid = hyper_LR
                            quantifier = EMQ(LogisticRegression())
                        elif method == 'PACC':
                            param_grid = hyper_LR
                            quantifier = PACC(LogisticRegression())
                        elif method == 'PACC-kfcv':
                            param_grid = hyper_LR
                            quantifier = PACC(LogisticRegression(), val_split=10)
                        elif method == 'PCC':
                            param_grid = hyper_LR
                            quantifier = PCC(LogisticRegression())
                        elif method == 'ACC':
                            param_grid = hyper_LR
                            quantifier = ACC(LogisticRegression())
                        elif method == 'CC':
                            param_grid = hyper_LR
                            quantifier = CC(LogisticRegression())
                        elif method == 'HDy-OvA':
                            param_grid = {
                                'binary_quantifier__classifier__C': np.logspace(-4,4,9),
                                'binary_quantifier__classifier__class_weight': ['balanced', None]
                            } 
                            quantifier = OneVsAllAggregative(HDy(LogisticRegression()))
                        elif method == 'DM':
                            param_grid = {
                                'nbins': [5,10,15], 
                                'classifier__C': np.logspace(-4,4,9),
                                'classifier__class_weight': ['balanced', None]
                            }
                            quantifier = DistributionMatching(LogisticRegression())
                        else:
                            raise NotImplementedError('unknown method', method)

                        # model selection
                        data = qp.datasets.fetch_twitter(dataset, min_df=3, pickle=True, for_model_selection=True)

                        protocol = UPP(data.test, repeats=n_bags_val)
                        modsel = GridSearchQ(quantifier, param_grid, protocol, refit=False, n_jobs=-1, verbose=1, error=optim)

                        modsel.fit(data.training)
                        print(f'best params {modsel.best_params_}')
                        pickle.dump(modsel.best_params_, open(f'{result_dir}/{method}_{dataset}.hyper.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

                        quantifier = modsel.best_model()

                        if is_semeval:
                            semeval_trained = True
                    
                    else:
                        print(f'model selection for {dataset} already done; skipping')

                    data = qp.datasets.fetch_twitter(dataset, min_df=3, pickle=True, for_model_selection=False)
                    quantifier.fit(data.training)
                    protocol = UPP(data.test, repeats=n_bags_test)
                    report = qp.evaluation.evaluation_report(quantifier, protocol, error_metrics=['mae', 'mrae', 'kld'], verbose=True)
                    report.to_csv(result_path+'.dataframe')
                    means = report.mean()
                    csv.write(f'{method}\t{data.name}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{means["kld"]:.5f}\n')
                    csv.flush()

    df = pd.read_csv(result_path+'.csv', sep='\t')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["MAE", "MRAE"])
    print(pv)
