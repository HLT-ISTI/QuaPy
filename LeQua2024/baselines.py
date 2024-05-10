import argparse
import pickle
import os
import sys
from os.path import join

import numpy as np
from sklearn.linear_model import LogisticRegression as LR

from scripts.constants import SAMPLE_SIZE
from LeQua2024._lequa2024 import LEQUA2024_TASKS, fetch_lequa2024, LEQUA2024_ZENODO
from quapy.method.aggregative import *
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation as MLPE
import quapy.functional as F


# LeQua official baselines (under development!)
# =================================================================================

BINARY_TASKS = ['T1', 'T4']


def new_cls():
    return LR(n_jobs=-1, max_iter=3000)


lr_params = {
    'C': np.logspace(-4, 4, 9),
    'class_weight': [None, 'balanced']
}

def wrap_params(cls_params:dict, prefix:str):
    return {'__'.join([prefix, key]): val for key, val in cls_params.items()}



def baselines():

    q_params = wrap_params(lr_params, 'classifier')

    yield CC(new_cls()), "CC", q_params
    yield ACC(new_cls()), "ACC", q_params
    yield PCC(new_cls()), "PCC", q_params
    yield PACC(new_cls()), "PACC", q_params
    yield KDEyML(new_cls()), "KDEy-ML", {**q_params, 'bandwidth': np.linspace(0.01, 0.20, 20)}


def main(args):

    models_path = qp.util.create_if_not_exist(join('./models', args.task))
    hyperparams_path = qp.util.create_if_not_exist(join('./hyperparams', args.task))

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(hyperparams_path, exist_ok=True)

    qp.environ['SAMPLE_SIZE'] = SAMPLE_SIZE[args.task]

    train, gen_val, gen_test = fetch_lequa2024(task=args.task, data_home=args.datadir, merge_T3=True)

    # gen_test is None, since the true prevalence vectors for the test samples will be released
    # only after the competition ends

    print(f'number of classes: {len(train.classes_)}')
    print(f'number of training documents: {len(train)}')
    print(f'training prevalence: {F.strprev(train.prevalence())}')
    print(f'training matrix shape: {train.instances.shape}')

    for quantifier, q_name, param_grid in baselines():

        model_path = os.path.join(models_path, q_name + '.pkl')
        modelparams_path = os.path.join(hyperparams_path, q_name + '.pkl')
        if os.path.exists(model_path):
            print(f'a pickle for {q_name} exists already in {model_path}; skipping!')
            continue

        print(f'starting model fitting for {q_name}')

        if param_grid is not None:
            optimizer = qp.model_selection.GridSearchQ(
                quantifier,
                param_grid,
                protocol=gen_val,
                error=qp.error.mrae,
                refit=False,
                verbose=True,
                n_jobs=-1
            ).fit(train)
            print(f'{q_name} got MRAE={optimizer.best_score_:.5f} (hyper-params: {optimizer.best_params_})')
            quantifier = optimizer.best_model()
        else:
            quantifier.fit(train)

        print(f'saving model in {model_path}')
        pickle.dump(quantifier, open(model_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(quantifier.get_params(), open(modelparams_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LeQua2024 baselines')
    parser.add_argument('task', metavar='TASK', type=str, choices=LEQUA2024_TASKS,
                        help=f'Code of the task; available ones are {LEQUA2024_TASKS}')
    parser.add_argument('datadir', metavar='DATA-PATH', type=str,
                        help='Path of the directory containing LeQua 2024 data (default is ./data)',
                        default='./data')
    args = parser.parse_args()

    main(args)
