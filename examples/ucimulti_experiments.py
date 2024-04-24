import pickle
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

import quapy as qp
from quapy.method.aggregative import PACC, EMQ, KDEyML
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP
from pathlib import Path


SEED = 1


def newLR():
    return LogisticRegression(max_iter=3000)

# typical hyperparameters explored for Logistic Regression
logreg_grid = {
    'C': np.logspace(-3, 3, 7),
    'class_weight': ['balanced', None]
}

def wrap_hyper(classifier_hyper_grid:dict):
    return {'classifier__'+k:v for k, v in classifier_hyper_grid.items()}

METHODS = [
    ('PACC', PACC(newLR()), wrap_hyper(logreg_grid)),
    ('EMQ',  EMQ(newLR()), wrap_hyper(logreg_grid)),
    # ('KDEy-ML',  KDEyML(newLR()), {**wrap_hyper(logreg_grid), **{'bandwidth': np.linspace(0.01, 0.2, 20)}}),
]


def show_results(result_path):
    import pandas as pd
    df = pd.read_csv(result_path+'.csv', sep='\t')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["MAE", "MRAE"], margins=True)
    print(pv)


if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = 500
    qp.environ['N_JOBS'] = -1
    n_bags_val = 250
    n_bags_test = 1000
    result_dir = f'results/ucimulti'

    os.makedirs(result_dir, exist_ok=True)

    global_result_path = f'{result_dir}/allmethods'
    with open(global_result_path + '.csv', 'wt') as csv:
        csv.write(f'Method\tDataset\tMAE\tMRAE\n')

    for method_name, quantifier, param_grid in METHODS:

        print('Init method', method_name)

        with open(global_result_path + '.csv', 'at') as csv:

            for dataset in qp.datasets.UCI_MULTICLASS_DATASETS:

                if dataset in []:
                    continue

                print('init', dataset)

                local_result_path = os.path.join(Path(global_result_path).parent, method_name + '_' + dataset + '.dataframe')

                if os.path.exists(local_result_path):
                    print(f'result file {local_result_path} already exist; skipping')
                    report = qp.util.load_report(local_result_path)

                else:
                    with qp.util.temp_seed(SEED):

                        data = qp.datasets.fetch_UCIMulticlassDataset(dataset, verbose=True)

                        # model selection
                        train, test = data.train_test
                        train, val = train.split_stratified(random_state=SEED)

                        protocol = UPP(val, repeats=n_bags_val)
                        modsel = GridSearchQ(
                            quantifier, param_grid, protocol, refit=True, n_jobs=-1, verbose=1, error='mae'
                        )

                        try:
                            modsel.fit(train)

                            print(f'best params {modsel.best_params_}')
                            print(f'best score {modsel.best_score_}')

                            quantifier = modsel.best_model()
                        except:
                            print('something went wrong... trying to fit the default model')
                            quantifier.fit(train)

                        protocol = UPP(test, repeats=n_bags_test)
                        report = qp.evaluation.evaluation_report(
                            quantifier, protocol, error_metrics=['mae', 'mrae'], verbose=True
                        )
                        report.to_csv(local_result_path)

                means = report.mean(numeric_only=True)
                csv.write(f'{method_name}\t{dataset}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\n')
                csv.flush()

    show_results(global_result_path)