import pickle
import os

from sklearn.linear_model import LogisticRegression

from distribution_matching.commons import METHODS, new_method, show_results

import quapy as qp
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP


SEED = 1


if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = 500
    qp.environ['N_JOBS'] = -1
    n_bags_val = 250
    n_bags_test = 1000
    for optim in ['mae', 'mrae']:
        result_dir = f'results/ucimulti/{optim}'

        os.makedirs(result_dir, exist_ok=True)

        for method in METHODS:

            print('Init method', method)

            global_result_path = f'{result_dir}/{method}'

            if not os.path.exists(global_result_path + '.csv'):
                with open(global_result_path + '.csv', 'wt') as csv:
                    csv.write(f'Method\tDataset\tMAE\tMRAE\tKLD\n')

            with open(global_result_path + '.csv', 'at') as csv:

                for dataset in qp.datasets.UCI_MULTICLASS_DATASETS:

                    print('init', dataset)

                    local_result_path = global_result_path + '_' + dataset
                    if os.path.exists(local_result_path + '.dataframe'):
                        print(f'result file {local_result_path}.dataframe already exist; skipping')
                        continue

                    with qp.util.temp_seed(SEED):

                        param_grid, quantifier = new_method(method, max_iter=3000)

                        data = qp.datasets.fetch_UCIMulticlassDataset(dataset)

                        # model selection
                        train, test = data.train_test
                        train, val = train.split_stratified(random_state=SEED)

                        protocol = UPP(val, repeats=n_bags_val)
                        modsel = GridSearchQ(
                            quantifier, param_grid, protocol, refit=True, n_jobs=-1, verbose=1, error=optim
                        )

                        try:
                            modsel.fit(train)

                            print(f'best params {modsel.best_params_}')
                            print(f'best score {modsel.best_score_}')
                            pickle.dump(
                                (modsel.best_params_, modsel.best_score_,),
                                open(f'{local_result_path}.hyper.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

                            quantifier = modsel.best_model()
                        except:
                            print('something went wrong... trying to fit the default model')
                            quantifier.fit(train)
                            # quantifier = qp.method.aggregative.CC(LogisticRegression()).fit(train)


                        protocol = UPP(test, repeats=n_bags_test)
                        report = qp.evaluation.evaluation_report(quantifier, protocol, error_metrics=['mae', 'mrae', 'kld'],
                                                                verbose=True, n_jobs=-1)
                        report.to_csv(f'{local_result_path}.dataframe')
                        means = report.mean()
                        csv.write(f'{method}\t{data.name}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{means["kld"]:.5f}\n')
                        csv.flush()

        show_results(global_result_path)
