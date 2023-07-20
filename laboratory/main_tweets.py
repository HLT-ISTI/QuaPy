import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import sys
import pandas as pd

import quapy as qp
from quapy.method.aggregative import DistributionMatching
from method_kdey import KDEy
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP


if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = 100
    qp.environ['N_JOBS'] = -1
    method = 'KDE'
    param = 0.1
    target = 'max_likelihood'
    div = 'topsoe'
    method_identifier = f'{method}_modsel_{div if target=="min_divergence" else target}'

    os.makedirs('results', exist_ok=True)
    result_path = f'results/{method_identifier}.csv'

    #if os.path.exists(result_path):
    #    print('Result already exit. Nothing to do')
    #    sys.exit(0)

    with open(result_path, 'wt') as csv:
        csv.write(f'Method\tDataset\tMAE\tMRAE\n')

        for dataset in qp.datasets.TWITTER_SENTIMENT_DATASETS_TEST:
            print('init', dataset)

            data = qp.datasets.fetch_twitter(dataset, min_df=3, pickle=True, for_model_selection=True)

            if method == 'KDE':
                param_grid = {'bandwidth': np.linspace(0.001, 0.2, 21)}
                model = KDEy(LogisticRegression(), divergence=div, bandwidth=param, engine='sklearn', target=target)
            else:
                raise NotImplementedError('unknown method')

            protocol = UPP(data.test, repeats=100)
            modsel = GridSearchQ(model, param_grid, protocol, refit=False, n_jobs=-1, verbose=1)

            modsel.fit(data.training)
            print(f'best params {modsel.best_params_}')

            quantifier = modsel.best_model()

            data = qp.datasets.fetch_twitter(dataset, min_df=3, pickle=True, for_model_selection=False)
            quantifier.fit(data.training)
            protocol = UPP(data.test, repeats=100)
            report = qp.evaluation.evaluation_report(quantifier, protocol, error_metrics=['mae', 'mrae'], verbose=True)
            means = report.mean()
            csv.write(f'{method_identifier}\t{data.name}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\n')
            csv.flush()

    df = pd.read_csv(result_path, sep='\t')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["MAE", "MRAE"])
    print(pv)
