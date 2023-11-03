import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import quapy as qp
from distribution_matching.commons import show_results
from method_kdey import KDEy
from quapy.method.aggregative import DistributionMatching


SEED=1

def task(val):
    print('job-init', val)
    train, val_gen, test_gen = qp.datasets.fetch_lequa2022('T1B')

    with qp.util.temp_seed(SEED):
        if method=='KDEy-ML':
            quantifier = KDEy(LogisticRegression(), target='max_likelihood', val_split=10, bandwidth=val)
        elif method == 'DM-HD':
            quantifier = DistributionMatching(LogisticRegression(), val_split=10, nbins=val, divergence='HD')

        quantifier.fit(train)
        report = qp.evaluation.evaluation_report(
            quantifier, protocol=test_gen, error_metrics=['mae', 'mrae', 'kld'], verbose=True)
        return report


if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = qp.datasets.LEQUA2022_SAMPLE_SIZE['T1B']
    qp.environ['N_JOBS'] = -1
    result_dir = f'results/lequa/T1B/sensibility'

    os.makedirs(result_dir, exist_ok=True)

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
                means = report.mean()
                local_result_path = global_result_path + '_T1B' + (f'_{val:.3f}' if isinstance(val, float) else f'{val}')
                report.to_csv(f'{local_result_path}.dataframe')
                csv.write(f'{method}\tLeQua-T1B\t{val}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{means["kld"]:.5f}\n')
                csv.flush()

        show_results(global_result_path)
