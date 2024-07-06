import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import quapy as qp
from distribution_matching.commons import show_results
from distribution_matching.method.kdey import KDEyML
from quapy.method.aggregative import DistributionMatching
from quapy.protocol import UPP


SEED=1

def task(val):
    print('job-init', dataset, val)

    with qp.util.temp_seed(SEED):
        if method == 'KDEy-ML':
            quantifier = KDEyML(LogisticRegression(n_jobs=-1), val_split=10, bandwidth=val)
        elif method == 'DM-HD':
            quantifier = DistributionMatching(LogisticRegression(n_jobs=-1), val_split=10, nbins=val, divergence='HD',
                                              n_jobs=-1)

        quantifier.fit(data.training)
        protocol = UPP(data.test, repeats=n_bags_test)
        report = qp.evaluation.evaluation_report(quantifier, protocol, error_metrics=['mae', 'mrae', 'kld'],
                                                 verbose=True, n_jobs=-1)
        return report


if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = 500
    qp.environ['N_JOBS'] = -1
    n_bags_val = 250
    n_bags_test = 1000
    result_dir = f'results/ucimulti/sensibility'

    os.makedirs(result_dir, exist_ok=True)

    for dataset in qp.datasets.UCI_MULTICLASS_DATASETS:

        data = qp.datasets.fetch_UCIMulticlassDataset(dataset)

        for method, param, grid in [
            ('KDEy-ML', 'Bandwidth', np.linspace(0.01, 0.2, 20)),
            ('DM-HD', 'nbins', list(range(2, 10)) + list(range(10, 34, 2)))
        ]:

            global_result_path = f'{result_dir}/{method}'

            if not os.path.exists(global_result_path+'.csv'):
                with open(global_result_path+'.csv', 'wt') as csv:
                    csv.write(f'Method\tDataset\t{param}\tMAE\tMRAE\tKLD\n')

            reports = qp.util.parallel(task, grid, n_jobs=-1)
            with open(global_result_path + '.csv', 'at') as csv:
                for val, report in zip(grid, reports):
                    local_result_path = global_result_path + '_' + dataset + (f'_{val:.3f}' if isinstance(val, float) else f'{val}')
                    report.to_csv(f'{local_result_path}.dataframe')
                    csv.write(f'{method}\t{dataset}\t{val}\t{report["mae"].mean():.5f}\t{report["mrae"].mean():.5f}\t{report["kld"].mean():.5f}\n')
                    csv.flush()

            show_results(global_result_path)
