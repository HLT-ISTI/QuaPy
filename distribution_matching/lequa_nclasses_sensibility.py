import pickle
import numpy as np
import os
from os.path import join
import pandas as pd
from quapy.protocol import UPP
from quapy.data import LabelledCollection
from distribution_matching.commons import METHODS, new_method, show_results
import quapy as qp


SEED=1


def extract_classes(data:LabelledCollection, classes):
    X, y = data.Xy
    counts = data.counts()
    Xs, ys = [], []
    for class_i in classes:
        Xs.append(X[y==class_i])
        ys.append([class_i]*counts[class_i])
    Xs = np.concatenate(Xs)
    ys = np.concatenate(ys)
    return LabelledCollection(Xs, ys, classes=classes
                              )

def task(nclasses):
    in_classes = np.arange(0, nclasses)
    train = extract_classes(train_pool, classes=in_classes)
    test = extract_classes(test_pool, classes=in_classes)
    with qp.util.temp_seed(SEED):
        hyper, quantifier = new_method(method)
        quantifier.set_params(classifier__C=1, classifier__class_weight='balanced')
        hyper = {h:v for h,v in hyper.items() if not h.startswith('classifier__')}
        tr, va = train.split_stratified(random_state=SEED)
        quantifier = qp.model_selection.GridSearchQ(quantifier, hyper, UPP(va), optim).fit(tr)
        report = qp.evaluation.evaluation_report(quantifier, protocol=UPP(test), error_metrics=['mae', 'mrae', 'kld'], verbose=True)
        return report


# only the quantifier-dependent hyperparameters are explored; the classifier is a LR with default parameters
if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = qp.datasets.LEQUA2022_SAMPLE_SIZE['T1B']
    qp.environ['N_JOBS'] = -1


    for optim in ['mae']: #, 'mrae']:

        result_dir = f'results/lequa/nclasses/{optim}'
        os.makedirs(result_dir, exist_ok=True)

        for method in ['DM', 'EMQ', 'KDEy-ML']: # 'KDEy-ML', 'KDEy-DMhd3']:

            result_path = join(result_dir, f'{method}.csv')
            if os.path.exists(result_path): continue

            train_orig, _, _ = qp.datasets.fetch_lequa2022('T1B')

            train_pool, test_pool = train_orig.split_stratified(0.5, random_state=SEED)
            arange_classes = np.arange(2, train_orig.n_classes + 1)
            reports = qp.util.parallel(task, arange_classes, n_jobs=-1)
            with open(result_path, 'at') as csv:
                csv.write(f'Method\tDataset\tnClasses\tMAE\tMRAE\tKLD\n')
                for num_classes, report in zip(arange_classes, reports):
                    means = report.mean()
                    report_result_path = join(result_dir, f'{method}_{num_classes}')+'.dataframe'
                    report.to_csv(report_result_path)
                    csv.write(f'{method}\tLeQua-T1B\t{num_classes}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{means["kld"]:.5f}\n')
                    csv.flush()

            means = report.mean()
            print(means)

