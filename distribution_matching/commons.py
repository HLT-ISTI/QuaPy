import numpy as np
import pandas as pd
from distribution_matching.method_kdey import KDEy
from distribution_matching.method_kdey_closed import KDEyclosed
from distribution_matching.method_kdey_closed_efficient_correct import KDEyclosed_efficient_corr
from quapy.method.aggregative import EMQ, CC, PCC, DistributionMatching, PACC, HDy, OneVsAllAggregative, ACC
from distribution_matching.method_dirichlety import DIRy
from sklearn.linear_model import LogisticRegression
from distribution_matching.method_kdey_closed_efficient import KDEyclosed_efficient

# the full list of methods tested in the paper (reported in the appendix)
METHODS  = ['ACC', 'PACC', 'HDy-OvA', 'DM-T', 'DM-HD', 'KDEy-HD', 'DM-CS', 'KDEy-CS',  'DIR', 'EMQ', 'EMQ-BCTS', 'KDEy-ML']

# uncomment this other list for the methods shown in the body of the paper (the other methods are not comparable in performance)
#METHODS  = ['PACC',  'DM-T', 'DM-HD', 'KDEy-HD', 'DM-CS', 'KDEy-CS',  'EMQ', 'KDEy-ML']

BIN_METHODS = [x.replace('-OvA', '') for x in METHODS]


hyper_LR = {
    'classifier__C': np.logspace(-3,3,7),
    'classifier__class_weight': ['balanced', None]
}

hyper_kde = {
    'bandwidth': np.linspace(0.01, 0.2, 20)
}

nbins_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 64]

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
    elif method in ['KDEy-HD']:
        param_grid = {**hyper_kde, **hyper_LR}
        quantifier = KDEy(lr, target='min_divergence', divergence='HD', montecarlo_trials=10000, val_split=10)
    elif method == 'KDEy-CS':
        param_grid = {**hyper_kde, **hyper_LR}
        quantifier = KDEyclosed_efficient_corr(lr, val_split=10)
    elif method == 'KDEy-ML':
        param_grid = {**hyper_kde, **hyper_LR}
        quantifier = KDEy(lr, target='max_likelihood', val_split=10)
    elif method == 'DIR':
        param_grid = hyper_LR
        quantifier = DIRy(lr)
    elif method == 'EMQ':
        param_grid = hyper_LR
        quantifier = EMQ(lr)
    elif method == 'EMQ-BCTS':
        method_params = {'exact_train_prev': [False], 'recalib': ['bcts']}
        param_grid = {**method_params, **hyper_LR}
        quantifier = EMQ(lr)
    elif method == 'HDy':
        param_grid = hyper_LR
        quantifier = HDy(lr)
    elif method == 'HDy-OvA':
        param_grid = {'binary_quantifier__' + key: val for key, val in hyper_LR.items()}
        quantifier = OneVsAllAggregative(HDy(lr))
    elif method == 'DM-T':
        method_params = {
            'nbins': nbins_range,
            'val_split': [10],
            'divergence': ['topsoe']
        }
        param_grid = {**method_params, **hyper_LR}
        quantifier = DistributionMatching(lr)
    elif method == 'DM-HD':
        method_params = {
            'nbins': nbins_range,
            'val_split': [10],
            'divergence': ['HD']
        }
        param_grid = {**method_params, **hyper_LR}
        quantifier = DistributionMatching(lr)
    elif method == 'DM-CS':
        method_params = {
            'nbins': nbins_range,
            'val_split': [10],
            'divergence': ['CS']
        }
        param_grid = {**method_params, **hyper_LR}
        quantifier = DistributionMatching(lr)
    else:
        raise NotImplementedError('unknown method', method)

    return param_grid, quantifier


def show_results(result_path):
    df = pd.read_csv(result_path+'.csv', sep='\t')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["MAE", "MRAE"])
    print(pv)

