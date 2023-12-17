from sklearn.linear_model import LogisticRegression
import os
import sys
import pandas as pd

import quapy as qp
from method.aggregative import DistributionMatching
from distribution_matching.method.method_kdey import KDEy
from protocol import UPP


if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = 100
    qp.environ['N_JOBS'] = -1
    method = 'KDE'
    param = 0.1
    div = 'topsoe'
    method_identifier = f'{method}_{param}_{div}'

    # generates tuples (dataset, method, method_name)
    # (the dataset is needed for methods that process the dataset differently)
    def gen_methods():

        for dataset in qp.datasets.TWITTER_SENTIMENT_DATASETS_TEST:

            data = qp.datasets.fetch_twitter(dataset, min_df=3, pickle=True)

            if method == 'KDE':
                kdey = KDEy(LogisticRegression(), divergence=div, bandwidth=param, engine='sklearn')
                yield data, kdey, method_identifier

            elif method == 'DM':
                dm = DistributionMatching(LogisticRegression(), divergence=div, nbins=param)
                yield data, dm, method_identifier

            else:
                raise NotImplementedError('unknown method')

    os.makedirs('results', exist_ok=True)
    result_path = f'results/{method_identifier}.csv'

    if os.path.exists(result_path):
        print('Result already exit. Nothing to do')
        sys.exit(0)

    with open(result_path, 'wt') as csv:
        csv.write(f'Method\tDataset\tMAE\tMRAE\n')
        for data, quantifier, quant_name in gen_methods():
            quantifier.fit(data.training)
            protocol = UPP(data.mixture, repeats=100)
            report = qp.evaluation.evaluation_report(quantifier, protocol, error_metrics=['mae', 'mrae'], verbose=True)
            means = report.mean()
            csv.write(f'{quant_name}\t{data.name}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\n')
            csv.flush()

    df = pd.read_csv(result_path, sep='\t')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["MAE", "MRAE"])
    print(pv)
