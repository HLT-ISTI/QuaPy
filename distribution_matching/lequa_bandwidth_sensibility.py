import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import pandas as pd
import quapy as qp
from method_kdey import KDEy


SEED=1

def task(bandwidth):
    print('job-init', dataset, bandwidth)
    train, val_gen, test_gen = qp.datasets.fetch_lequa2022(dataset)

    with qp.util.temp_seed(SEED):
        quantifier = KDEy(LogisticRegression(), target='max_likelihood', val_split=10, bandwidth=bandwidth)
        quantifier.fit(train)
        report = qp.evaluation.evaluation_report(
            quantifier, protocol=test_gen, error_metrics=['mae', 'mrae', 'kld'], verbose=True)
        return report


if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = qp.datasets.LEQUA2022_SAMPLE_SIZE['T1B']
    qp.environ['N_JOBS'] = -1
    result_dir = f'results_lequa_sensibility'

    os.makedirs(result_dir, exist_ok=True)

    method = 'KDEy-MLE'

    global_result_path = f'{result_dir}/{method}'

    if not os.path.exists(global_result_path+'.csv'):
        with open(global_result_path+'.csv', 'wt') as csv:
            csv.write(f'Method\tDataset\tBandwidth\tMAE\tMRAE\tKLD\n')

    dataset = 'T1B'
    bandwidths = np.linspace(0.01, 0.2, 20)

    reports = qp.util.parallel(task, bandwidths, n_jobs=-1)
    with open(global_result_path + '.csv', 'at') as csv:
        for bandwidth, report in zip(bandwidths, reports):
            means = report.mean()
            local_result_path = global_result_path + '_' + dataset + f'_{bandwidth:.3f}'
            report.to_csv(f'{local_result_path}.dataframe')
            csv.write(f'{method}\tLeQua-T1B\t{bandwidth}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{means["kld"]:.5f}\n')
            csv.flush()

    df = pd.read_csv(global_result_path + '.csv', sep='\t')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["MAE", "MRAE"])
    print(pv)
