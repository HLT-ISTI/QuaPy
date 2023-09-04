import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import sys
import pandas as pd

import quapy as qp
from quapy.method.aggregative import EMQ, DistributionMatching, PACC, ACC, CC, PCC, HDy, OneVsAllAggregative
from method_kdey import KDEy
from method_dirichlety import DIRy
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP

SEED=1

if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = 100
    qp.environ['N_JOBS'] = -1
    n_bags_val = 250
    n_bags_test = 1000
    result_dir = f'results_tweet_sensibility'

    os.makedirs(result_dir, exist_ok=True)

    method = 'KDEy-MLE'
        
    global_result_path = f'{result_dir}/{method}'
    
    if not os.path.exists(global_result_path+'.csv'):
        with open(global_result_path+'.csv', 'wt') as csv:
            csv.write(f'Method\tDataset\tBandwidth\tMAE\tMRAE\tKLD\n')    

    with open(global_result_path+'.csv', 'at') as csv:
        # four semeval dataset share the training, so it is useless to optimize hyperparameters four times;
        # this variable controls that the mod sel has already been done, and skip this otherwise
        semeval_trained = False

        for bandwidth in np.linspace(0.01, 0.2, 20):                        
            for dataset in qp.datasets.TWITTER_SENTIMENT_DATASETS_TEST:
                print('init', dataset)

                local_result_path = global_result_path + '_' + dataset + f'_{bandwidth:.3f}'
                
                with qp.util.temp_seed(SEED):

                    data = qp.datasets.fetch_twitter(dataset, min_df=3, pickle=True, for_model_selection=False)
                    quantifier = KDEy(LogisticRegression(), target='max_likelihood', val_split=10, bandwidth=bandwidth)
                    quantifier.fit(data.training)
                    protocol = UPP(data.test, repeats=n_bags_test)
                    report = qp.evaluation.evaluation_report(quantifier, protocol, error_metrics=['mae', 'mrae', 'kld'], verbose=True)
                    report.to_csv(f'{local_result_path}.dataframe')
                    means = report.mean()
                    csv.write(f'{method}\t{data.name}\t{bandwidth}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{means["kld"]:.5f}\n')
                    csv.flush()

    df = pd.read_csv(global_result_path+'.csv', sep='\t')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["MAE", "MRAE"])
    print(pv)
