import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import sys
import pandas as pd

import quapy as qp
from quapy.method.aggregative import DistributionMatching
from method_kdey import KDEy
from quapy.model_selection import GridSearchQ


if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = qp.datasets.LEQUA2022_SAMPLE_SIZE['T1B']
    qp.environ['N_JOBS'] = -1
    method = 'KDE'
    param = 0.1
    div = 'topsoe'
    method_identifier = f'{method}_modsel_{div}'

    os.makedirs('results', exist_ok=True)
    result_path = f'results_LequaT2B/{method_identifier}.csv'

    #if os.path.exists(result_path):
    #    print('Result already exit. Nothing to do')
    #    sys.exit(0)

    with open(result_path, 'wt') as csv:
        csv.write(f'Method\tDataset\tMAE\tMRAE\n')

        dataset = 'T1B'
        train, val_gen, test_gen = qp.datasets.fetch_lequa2022(dataset)

        if method == 'KDE':
            param_grid = {'bandwidth': np.linspace(0.001, 0.1, 11)}
            model = KDEy(LogisticRegression(), divergence=div, bandwidth=param, engine='sklearn')
        else:
            raise NotImplementedError('unknown method')

        modsel = GridSearchQ(model, param_grid, protocol=val_gen, refit=False, n_jobs=-1, verbose=1)

        modsel.fit(train)
        print(f'best params {modsel.best_params_}')

        quantifier = modsel.best_model()

        report = qp.evaluation.evaluation_report(quantifier, protocol=test_gen, error_metrics=['mae', 'mrae'], verbose=True)
        means = report.mean()
        csv.write(f'{method}\tLeQua-{dataset}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\n')
        csv.flush()

    df = pd.read_csv(result_path, sep='\t')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["MAE", "MRAE"])
    print(pv)
