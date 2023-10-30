import numpy as np
import pandas as pd
from distribution_matching.method_kdey import KDEy
from distribution_matching.method_kdey_closed import KDEyclosed
from distribution_matching.method_kdey_closed_efficient_correct import KDEyclosed_efficient_corr
from quapy.method.aggregative import EMQ, CC, PCC, DistributionMatching, PACC, HDy, OneVsAllAggregative, ACC
from distribution_matching.method_dirichlety import DIRy
from sklearn.linear_model import LogisticRegression
from method_kdey_closed_efficient import KDEyclosed_efficient

METHODS  = ['ACC', 'PACC', 'HDy-OvA', 'DIR', 'DM-T', 'DM-HD', 'KDEy-DMhd3', 'DM-CS', 'KDEy-closed++', 'EMQ', 'KDEy-ML'] #['ACC', 'PACC', 'HDy-OvA', 'DIR', 'DM', 'KDEy-DMhd3', 'KDEy-closed++', 'EMQ', 'KDEy-ML'] #, 'KDEy-DMhd2'] #, 'KDEy-DMhd2', 'DM-HD'] 'KDEy-DMjs', 'KDEy-DM', 'KDEy-ML+', 'KDEy-DMhd3+', 'EMQ-C',
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
    elif method == 'KDEy-closed':
        method_params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
        param_grid = {**method_params, **hyper_LR}
        quantifier = KDEyclosed(lr, val_split=10)
    elif method == 'KDEy-closed+':
        method_params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
        param_grid = {**method_params, **hyper_LR}
        quantifier = KDEyclosed_efficient(lr, val_split=10)
    elif method == 'KDEy-closed++':
        method_params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
        param_grid = {**method_params, **hyper_LR}
        quantifier = KDEyclosed_efficient_corr(lr, val_split=10)
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
    elif method == 'EMQ-C':
        method_params = {'exact_train_prev': [False], 'recalib': ['bcts']}
        param_grid = {**method_params, **hyper_LR}
        quantifier = EMQ(lr)
    elif method == 'HDy':
        param_grid = hyper_LR
        quantifier = HDy(lr)
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
    elif method == 'DM-T':
        method_params = {
            'nbins': [2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,32,64],
            'val_split': [10],
            'divergence': ['topsoe']
        }
        param_grid = {**method_params, **hyper_LR}
        quantifier = DistributionMatching(lr)
    elif method == 'DM-HD':
        method_params = {
            'nbins': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 64],
            'val_split': [10],
            'divergence': ['HD']
        }
        param_grid = {**method_params, **hyper_LR}
        quantifier = DistributionMatching(lr)
    elif method == 'DM-CS':
        method_params = {
            'nbins': [2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,32,64],
            'val_split': [10],
            'divergence': ['CS']
        }
        param_grid = {**method_params, **hyper_LR}
        quantifier = DistributionMatching(lr)

    # experimental
    elif method in ['KDEy-DMkld']:
        method_params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
        param_grid = {**method_params, **hyper_LR}
        quantifier = KDEy(lr, target='min_divergence', divergence='KLD', montecarlo_trials=5000, val_split=10)
    # elif method in ['KDEy-DMhd']:
    #     The code to reproduce this run is commented in the min_divergence target, I think it was incorrect...
    #     method_params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
    #     param_grid = {**method_params, **hyper_LR}
    #     quantifier = KDEy(lr, target='min_divergence', divergence='HD', montecarlo_trials=5000, val_split=10)
    elif method in ['KDEy-DMhd2']:
        method_params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
        param_grid = {**method_params, **hyper_LR}
        quantifier = KDEy(lr, target='min_divergence_uniform', divergence='HD', montecarlo_trials=5000, val_split=10)
    elif method in ['KDEy-DMjs']:
        method_params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
        param_grid = {**method_params, **hyper_LR}
        quantifier = KDEy(lr, target='min_divergence_uniform', divergence='JS', montecarlo_trials=5000, val_split=10)
    elif method in ['KDEy-DMhd3']:
        # I have realized that there was an error. I am sampling from the validation distribution (V) and not from the
        # test distribution (T) just because the validation can be sampled in fit only once and pre-computed densities
        # can be stored. This means that the reference distribution is V and not T. Then I have found that an
        # f-divergence is defined as D(p||q) \int_{R^n}q(x)f(p(x)/q(x))dx = E_{x~q}[f(p(x)/q(x))], so if I am sampling
        # V then I am computing D(T||V) (and not D(V||T) as I thought).
        method_params = {'bandwidth': np.linspace(0.01, 0.2, 20)}
        param_grid = {**method_params, **hyper_LR}
        quantifier = KDEy(lr, target='min_divergence', divergence='HD', montecarlo_trials=5000, val_split=10)
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

