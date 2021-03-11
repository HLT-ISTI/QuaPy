from glob import glob
import pickle
import numpy as np

results = './results'

method_choices = {}
for file in glob(f'{results}/*'):
    hyper = pickle.load(open(file, 'rb'))[-1]
    if hyper:
        dataset,method,optim = file.split('/')[-1].split('-')
        key = str(hyper)
        if method not in method_choices:
            method_choices[method] = {}
        if key not in method_choices[method]:
            method_choices[method][key] = 0
        method_choices[method][key] = method_choices[method][key]+1

for method, hyper_count_dict in method_choices.items():
    hyper, counts = zip(*list(hyper_count_dict.items()))
    order = np.argsort(counts)
    counts = np.asarray(counts)[order][::-1]
    hyper = np.asarray(hyper)[order][::-1]
    print(method)
    for hyper_i, count_i in zip(hyper, counts):
        print('\t', hyper_i, count_i)
