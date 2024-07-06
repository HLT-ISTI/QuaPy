import pickle
import numpy as np
import os
import pandas as pd
from distribution_matching.commons import METHODS, BIN_METHODS, new_method, show_results

import quapy as qp
from quapy.model_selection import GridSearchQ


if __name__ == '__main__':

    for task in ['T1A', 'T1B']:
        qp.environ['SAMPLE_SIZE'] = qp.datasets.LEQUA2022_SAMPLE_SIZE[task]
        qp.environ['N_JOBS'] = -1
        for optim in ['mae', 'mrae']:

            result_dir = f'results/lequa/{task}/{optim}'

            os.makedirs(result_dir, exist_ok=True)

            for method in (METHODS if task=='T1B' else BIN_METHODS):

                print('Init method', method)

                result_path = f'{result_dir}/{method}'

                if os.path.exists(result_path+'.csv'):
                    print(f'file {result_path}.csv already exist; skipping')
                    continue

                with open(result_path+'.csv', 'wt') as csv:
                    csv.write(f'Method\tDataset\tMAE\tMRAE\tKLD\n')

                    dataset = task
                    train, val_gen, test_gen = qp.datasets.fetch_lequa2022(dataset)
                    print(f'init {dataset} #instances: {len(train)}')
                    param_grid, quantifier = new_method(method)

                    if param_grid is not None:
                        modsel = GridSearchQ(quantifier, param_grid, protocol=val_gen, refit=False, n_jobs=-1, verbose=1, error=optim)

                        modsel.fit(train)
                        print(f'best params {modsel.best_params_}')
                        print(f'best score {modsel.best_score_}')
                        pickle.dump(
                            (modsel.best_params_, modsel.best_score_,),
                            open(f'{result_path}.hyper.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

                        quantifier = modsel.best_model()
                    else:
                        print('debug mode... skipping model selection')
                        quantifier.fit(train)

                    report = qp.evaluation.evaluation_report(
                        quantifier, protocol=test_gen, error_metrics=['mae', 'mrae', 'kld'],
                        verbose=True, verbose_error=optim[1:], n_jobs=-1
                    )
                    report.to_csv(result_path+'.dataframe')
                    csv.write(f'{method}\tLeQua-{task}\t{report["mae"].mean():.5f}\t{report["mrae"].mean():.5f}\t{report["kld"].mean():.5f}\n')
                    csv.flush()

        show_results(result_path)
