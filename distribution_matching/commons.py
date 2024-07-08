import numpy as np
import pandas as pd

from distribution_matching.method.edy import EDy
from distribution_matching.method.kdey import KDEyCS, KDEyHD, KDEyML
from quapy.method.aggregative import EMQ, CC, PCC, DistributionMatching, PACC, HDy, OneVsAllAggregative, ACC, \
    MedianSweep, MedianSweep2
from distribution_matching.method.dirichlety import DIRy
from sklearn.linear_model import LogisticRegression

# set to True to get the full list of methods tested in the paper (reported in the appendix)
# set to False to get the reduced list (shown in the body of the paper)
FULL_METHOD_LIST = True

if FULL_METHOD_LIST:
    ADJUSTMENT_METHODS = ['ACC+', 'PACC+']
    DISTR_MATCH_METHODS = ['EDy+', 'HDy-OvA', 'DM-T', 'DM-HD', 'KDEy-HD', 'DM-CS', 'KDEy-CS']
    MAX_LIKE_METHODS = ['DIR', 'EMQ', 'EMQ-BCTS', 'KDEy-ML']
else:
    ADJUSTMENT_METHODS = ['PACC+']
    DISTR_MATCH_METHODS = ['EDy+', 'DM-T', 'DM-HD', 'KDEy-HD',  'DM-CS', 'KDEy-CS']
    MAX_LIKE_METHODS = ['EMQ', 'KDEy-ML']


# list of methods to consider
METHODS = ADJUSTMENT_METHODS + DISTR_MATCH_METHODS + MAX_LIKE_METHODS
BIN_ADJUSTMENT_METHODS = ADJUSTMENT_METHODS + ['MS', 'MS2']
BIN_METHODS = BIN_ADJUSTMENT_METHODS + DISTR_MATCH_METHODS + MAX_LIKE_METHODS
BIN_METHODS = [x.replace('-OvA', '') for x in BIN_METHODS]


# common hyperparameterss
hyper_LR = {
    'classifier__C': np.logspace(-3, 3, 7),
    'classifier__class_weight': ['balanced', None]
}

hyper_kde = {
    'bandwidth': np.linspace(0.01, 0.2, 20)
}

nbins_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 64]


# instances a new quantifier based on a string name
def new_method(method, **lr_kwargs):
    lr = LogisticRegression(**lr_kwargs)

    if method == 'CC':
        param_grid = hyper_LR
        quantifier = CC(lr)
    elif method == 'PCC':
        param_grid = hyper_LR
        quantifier = PCC(lr)
    elif method == 'ACC+':
        param_grid = hyper_LR
        quantifier = ACC(lr)
    elif method == 'PACC+':
        param_grid = hyper_LR
        quantifier = PACC(lr)
    elif method == 'MS':
        param_grid = hyper_LR
        quantifier = MedianSweep(lr)
    elif method == 'MS2':
        param_grid = hyper_LR
        quantifier = MedianSweep2(lr)
    elif method in ['KDEy-HD']:
        param_grid = {**hyper_kde, **hyper_LR}
        quantifier = KDEyHD(lr)
    elif method == 'KDEy-CS':
        param_grid = {**hyper_kde, **hyper_LR}
        quantifier = KDEyCS(lr)
    elif method == 'KDEy-ML':
        param_grid = {**hyper_kde, **hyper_LR}
        quantifier = KDEyML(lr)
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
    elif method == 'EDy+':
        param_grid = {'distance': ['manhattan', 'euclidean'], **hyper_LR}
        quantifier = EDy(lr)
    else:
        raise NotImplementedError('unknown method', method)

    return param_grid, quantifier


def show_results(result_path):
    df = pd.read_csv(result_path+'.csv', sep='\t')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["MAE", "MRAE"])
    print(pv)

