import numpy as np
import pandas as pd
from distribution_matching.method_kdey import KDEy
from quapy.method.aggregative import EMQ, CC, PCC, DistributionMatching, PACC, HDy, OneVsAllAggregative, ACC
from distribution_matching.method_dirichlety import DIRy
from sklearn.linear_model import LogisticRegression


METHODS  = ['KDEy-DMjs', 'ACC', 'PACC', 'HDy-OvA', 'DIR', 'DM', 'KDEy-DM', 'EMQ', 'KDEy-ML'] #, 'KDEy-DMhd2'] #, 'KDEy-DMhd2', 'DM-HD']
BIN_METHODS = [x.replace('-OvA', '') for x in METHODS]


hyper_LR = {
    'classifier__C': np.logspace(-3,3,7),
    'classifier__class_weight': ['balanced', None]
} 

def new_method(method, **lr_kwargs):

    lr = LogisticRegression(**lr_kwargs)

    if method == 'CC':
        param_grid = hyper_LR
        quantifier = CC(lr)
    elif method == 'PCC':
        param_grid = hyper_LR
        quantifier = PCC(lr)
    elif method == 'ACC':
        param_grid = hyper_LR
        quantifier = ACC(lr)
    elif method == 'PACC':
        param_grid = hyper_LR
        quantifier = PACC(lr)
    elif method == 'KDEy-ML':
        method_params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
        param_grid = {**method_params, **hyper_LR}
        quantifier = KDEy(lr, target='max_likelihood', val_split=10)
    elif method in ['KDEy-DM']:
        method_params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
        param_grid = {**method_params, **hyper_LR}
        quantifier = KDEy(lr, target='min_divergence', divergence='l2', montecarlo_trials=5000, val_split=10)
    elif method == 'DIR':
        param_grid = hyper_LR
        quantifier = DIRy(lr)
    elif method == 'EMQ':
        param_grid = hyper_LR
        quantifier = EMQ(lr)
    elif method == 'HDy-OvA':
        param_grid = {'binary_quantifier__' + key: val for key, val in hyper_LR.items()}
        quantifier = OneVsAllAggregative(HDy(lr))
    elif method == 'DM':
        method_params = {
            'nbins': [4,8,16,32],
            'val_split': [10, 0.4],
            'divergence': ['HD', 'topsoe', 'l2']
        }
        param_grid = {**method_params, **hyper_LR}
        quantifier = DistributionMatching(lr)

    # experimental
    elif method in ['KDEy-DMkld']:
        method_params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
        param_grid = {**method_params, **hyper_LR}
        quantifier = KDEy(lr, target='min_divergence', divergence='KLD', montecarlo_trials=5000, val_split=10)
    elif method in ['KDEy-DMhd']:
        method_params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
        param_grid = {**method_params, **hyper_LR}
        quantifier = KDEy(lr, target='min_divergence', divergence='HD', montecarlo_trials=5000, val_split=10)
    elif method in ['KDEy-DMhd2']:
        method_params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
        param_grid = {**method_params, **hyper_LR}
        quantifier = KDEy(lr, target='min_divergence_uniform', divergence='HD', montecarlo_trials=5000, val_split=10)
    elif method in ['KDEy-DMjs']:
        method_params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
        param_grid = {**method_params, **hyper_LR}
        quantifier = KDEy(lr, target='min_divergence_uniform', divergence='JS', montecarlo_trials=5000, val_split=10)
    elif method == 'DM-HD':
        method_params = {
            'nbins': [4,8,16,32],
            'val_split': [10, 0.4],
        }
        param_grid = {**method_params, **hyper_LR}
        quantifier = DistributionMatching(lr, divergence='HD')

    else:
        raise NotImplementedError('unknown method', method)

    return param_grid, quantifier


def show_results(result_path):
    df = pd.read_csv(result_path+'.csv', sep='\t')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["MAE", "MRAE"])
    print(pv)

