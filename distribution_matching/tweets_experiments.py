import pickle
import os
import pandas as pd
from distribution_matching.commons import METHODS, new_method, show_results

import quapy as qp
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP

SEED=1

if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = 100
    qp.environ['N_JOBS'] = -1
    n_bags_val = 250
    n_bags_test = 1000
    optim = 'mae'
    result_dir = f'results/tweet/{optim}'

    os.makedirs(result_dir, exist_ok=True)

    for method in  METHODS:
        
        print('Init method', method)

        global_result_path = f'{result_dir}/{method}'
        
        if not os.path.exists(global_result_path+'.csv'):
            with open(global_result_path+'.csv', 'wt') as csv:
                csv.write(f'Method\tDataset\tMAE\tMRAE\tKLD\n')    

        with open(global_result_path+'.csv', 'at') as csv:
            # four semeval dataset share the training, so it is useless to optimize hyperparameters four times;
            # this variable controls that the mod sel has already been done, and skip this otherwise
            semeval_trained = False

            for dataset in qp.datasets.TWITTER_SENTIMENT_DATASETS_TEST:
                print('init', dataset)

                local_result_path = global_result_path + '_' + dataset
                if os.path.exists(local_result_path+'.dataframe'):
                    print(f'result file {local_result_path}.dataframe already exist; skipping')
                    continue 
                
                with qp.util.temp_seed(SEED):

                    is_semeval = dataset.startswith('semeval')

                    if not is_semeval or not semeval_trained:

                        param_grid, quantifier = new_method(method)

                        # model selection
                        data = qp.datasets.fetch_twitter(dataset, min_df=3, pickle=True, for_model_selection=True)

                        protocol = UPP(data.test, repeats=n_bags_val)
                        modsel = GridSearchQ(quantifier, param_grid, protocol, refit=False, n_jobs=-1, verbose=1, error=optim)

                        modsel.fit(data.training)
                        print(f'best params {modsel.best_params_}')
                        print(f'best score {modsel.best_score_}')
                        pickle.dump(
                            (modsel.best_params_, modsel.best_score_,), 
                            open(f'{local_result_path}.hyper.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

                        quantifier = modsel.best_model()

                        if is_semeval:
                            semeval_trained = True
                    
                    else:
                        print(f'model selection for {dataset} already done; skipping')

                    data = qp.datasets.fetch_twitter(dataset, min_df=3, pickle=True, for_model_selection=False)
                    quantifier.fit(data.training)
                    protocol = UPP(data.test, repeats=n_bags_test)
                    report = qp.evaluation.evaluation_report(quantifier, protocol, error_metrics=['mae', 'mrae', 'kld'], verbose=True)
                    report.to_csv(f'{local_result_path}.dataframe')
                    means = report.mean()
                    csv.write(f'{method}\t{data.name}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{means["kld"]:.5f}\n')
                    csv.flush()

    show_results(global_result_path)
