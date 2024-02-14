import argparse
import pickle
import os
from os.path import join
from sklearn.linear_model import LogisticRegression as LR

from scripts.constants import SAMPLE_SIZE
from LeQua2024._lequa2024 import LEQUA2024_TASKS, fetch_lequa2024
from quapy.method.aggregative import *
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation as MLPE
import quapy.functional as F


# LeQua official baselines
# =================================================================================

BINARY_TASKS = ['T1', 'T4']


def new_cls():
    return LR(n_jobs=-1)


lr_params = {
    'C': np.logspace(-3, 3, 7),
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
    yield EMQ(CalibratedClassifierCV(new_cls())), "SLD-Platt", wrap_params(wrap_params(lr_params, 'estimator'), 'classifier')
    yield EMQ(new_cls()), "SLD", q_params
    # yield EMQ(new_cls()), "SLD-BCTS", {**q_params, 'recalib': ['bcts'], 'val_split': [5]}
    yield MLPE(), "MLPE", None
    if args.task in BINARY_TASKS:
        yield MS2(new_cls()), "MedianSweep2", q_params


def main(args):

    models_path = qp.util.create_if_not_exist(join('./models', args.task))

    qp.environ['SAMPLE_SIZE'] = SAMPLE_SIZE[args.task]

    train, gen_val, gen_test = fetch_lequa2024(task=args.task, data_home=args.datadir, merge_T3=True)

    print(f'number of classes: {len(train.classes_)}')
    print(f'number of training documents: {len(train)}')
    print(f'training prevalence: {F.strprev(train.prevalence())}')
    print(f'training matrix shape: {train.instances.shape}')

    for quantifier, q_name, param_grid in baselines():

        model_path = os.path.join(models_path, q_name + '.pkl')
        if os.path.exists(model_path):
            print(f'a pickle for {q_name} exists already in {model_path}; skipping!')
            continue

        if param_grid is not None:
            quantifier = qp.model_selection.GridSearchQ(
                quantifier,
                param_grid,
                protocol=gen_val,
                error=qp.error.mrae,
                refit=False,
                verbose=True,
                n_jobs=-1
            ).fit(train)
            print(f'{q_name} got MRAE={quantifier.best_score_:.5f} (hyper-params: {quantifier.best_params_})')
            quantifier = quantifier.best_model()
        else:
            quantifier.fit(train)


        # valid_error = quantifier.best_score_

        # test_err = qp.evaluation.evaluate(quantifier, protocol=gen_test, error_metric='mrae', verbose=True)
        # print(f'method={q_name} got MRAE={test_err:.4f}')
        #
        # results.append((q_name, valid_error, test_err))


        print(f'saving model in {model_path}')
        pickle.dump(quantifier, open(model_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


    # print('\nResults')
    # print('Method\tValid-err\ttest-err')
    # for q_name, valid_error, test_err in results:
    #     print(f'{q_name}\t{valid_error:.4}\t{test_err:.4f}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LeQua2024 baselines')
    parser.add_argument('task', metavar='TASK', type=str, choices=LEQUA2024_TASKS,
                        help=f'Code of the task; available ones are {LEQUA2024_TASKS}')
    parser.add_argument('datadir', metavar='DATA-PATH', type=str,
                        help='Path of the directory containing LeQua 2024 data', default='./data')
    args = parser.parse_args()

    # def assert_file(filename):
    #     if not os.path.exists(os.path.join(args.datadir, filename)):
    #         raise FileNotFoundError(f'path {args.datadir} does not contain "{filename}"')
    #
    # assert_file('dev_prevalences.txt')
    # assert_file('training_data.txt')
    # assert_file('dev_samples')

    main(args)
