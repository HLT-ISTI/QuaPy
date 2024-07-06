from sklearn.linear_model import LogisticRegression
import os
import quapy as qp
import numpy as np
from distribution_matching.method.kdey import KDEyML, KDEyHD, KDEyCS
from time import time
import pickle
from quapy.method.aggregative import PACC, EMQ, DistributionMatching
from distribution_matching.method.edy import EDy

data = qp.datasets.fetch_UCIMulticlassLabelledCollection('poker_hand')
test, train = data.split_stratified(train_prop=500000, random_state=0)

train_q, train_cls = train.split_stratified(train_prop=500000)

classifier = LogisticRegression()
classifier.fit(*train_cls.Xy)

print(f'{len(train_cls)=}')
print(f'{len(train_q)=}')
print(f'{len(test)=}')
print(f'classes={train.n_classes}')
print(f'num features={train.X.shape[1]}')

start = np.log10(50)
end = np.log10(100000)
num_points = 10

log_space_n = np.logspace(start, end, num=num_points, dtype=int)

REPEATS=5

def methods():
    yield 'PACC', PACC(classifier=classifier)
    yield 'EMQ', EMQ(classifier=classifier)
    yield 'EDy', EDy(classifier=classifier)
    yield 'DM-10b', DistributionMatching(classifier=classifier, nbins=10)
    yield 'DM-50b', DistributionMatching(classifier=classifier, nbins=50)
    yield 'KDEy-ML', KDEyML(classifier=classifier, n_jobs=data.n_classes)
    yield 'KDEy-HD', KDEyHD(classifier=classifier, n_jobs=data.n_classes)
    yield 'KDEy-CS', KDEyCS(classifier=classifier, n_jobs=data.n_classes)


def train_quantifier(train, q_name):
    if q_name == 'EMQ':
        Q.fit(train, fit_classifier=False)
    else:
        Q.fit(train, fit_classifier=False, val_split=train)


if __name__ == '__main__':

    for q_name, Q in methods():
        print(f'starting: {q_name}')
        times_file = f'times/{q_name}.pkl'
        if os.path.exists(times_file):
            print(f'file {times_file} already exists; skipping')
            continue

        print('training times')
        tr_times = {}
        n_classes = data.n_classes
        uniform = [1/n_classes] * n_classes
        for n in log_space_n:
            print(f'training with n={n} ', end='')
            train_q_it = train_q.sampling(n, *uniform)
            tinit = time()
            for _ in range(REPEATS):
                train_quantifier(train_q_it, q_name)
            tend = (time()-tinit)/REPEATS
            if q_name=='EMQ':
                tend=0.0001
            print(f'took {tend:.6f} s')
            tr_times[n] = tend

        print('test times')
        train_q_10000 = train_q.sampling(10000, *uniform)
        train_quantifier(train_q_10000, q_name)

        te_times = {}
        for n in log_space_n:
            print(f'testing with n={n} ', end='')
            test_it = test.sampling(size=n)
            tinit = time()
            for _ in range(REPEATS):
                Q.quantify(test_it.X)
            tend = (time()-tinit)/REPEATS
            print(f'took {tend:.6f} s')
            te_times[n] = tend

        os.makedirs('times', exist_ok=True)
        pickle.dump((tr_times, te_times), open(times_file, 'wb'), pickle.HIGHEST_PROTOCOL)

