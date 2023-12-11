import numpy as np
from sklearn.linear_model import LogisticRegression
import os

import quapy as qp
from distribution_matching.commons import show_results
from quapy.method.aggregative import DMy
from distribution_matching.method.method_kdey import KDEy
from quapy.protocol import UPP

SEED=1

if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = 100
    qp.environ['N_JOBS'] = -1
    n_bags_val = 250
    n_bags_test = 1000
    result_dir = f'results/tweet/sensibility'

    os.makedirs(result_dir, exist_ok=True)

    for method, param, grid in [
        ('KDEy-ML', 'Bandwidth', np.linspace(0.01, 0.2, 20)),
        ('DM-HD', 'nbins', list(range(2,10)) + list(range(10,34,2)))
    ]:

        global_result_path = f'{result_dir}/{method}'

        if not os.path.exists(global_result_path+'.csv'):
            with open(global_result_path+'.csv', 'wt') as csv:
                csv.write(f'Method\tDataset\t{param}\tMAE\tMRAE\tKLD\n')

        with open(global_result_path+'.csv', 'at') as csv:
            for val in grid:
                for dataset in qp.datasets.TWITTER_SENTIMENT_DATASETS_TEST:
                    print('init', dataset)

                    local_result_path = global_result_path + '_' + dataset + (f'_{val:.3f}' if isinstance(val, float) else f'{val}')

                    with qp.util.temp_seed(SEED):

                        data = qp.datasets.fetch_twitter(dataset, min_df=3, pickle=True, for_model_selection=False)

                        if method == 'KDEy-ML':
                            quantifier = KDEy(LogisticRegression(n_jobs=-1), target='max_likelihood', val_split=10, bandwidth=val)
                        elif method == 'DM-HD':
                            quantifier = DMy(LogisticRegression(n_jobs=-1), val_split=10, nbins=val, divergence='HD', n_jobs=-1)
                        quantifier.fit(data.training)
                        protocol = UPP(data.test, repeats=n_bags_test)
                        report = qp.evaluation.evaluation_report(quantifier, protocol, error_metrics=['mae', 'mrae', 'kld'], verbose=True, n_jobs=-1)
                        report.to_csv(f'{local_result_path}.dataframe')
                        means = report.mean()
                        csv.write(f'{method}\t{data.name}\t{val}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{means["kld"]:.5f}\n')
                        csv.flush()

        show_results(global_result_path)
