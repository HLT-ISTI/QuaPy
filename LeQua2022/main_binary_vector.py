import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pandas as pd

import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import *
import quapy.functional as F
from data import load_binary_vectors
import os

path_binary_vector = './data/T1A'
result_path = os.path.join('results', 'T1A')  # binary - vector
os.makedirs(result_path, exist_ok=True)

train_file = os.path.join(path_binary_vector, 'public', 'training_vectors.txt')

train = LabelledCollection.load(train_file, load_binary_vectors)

nF = train.instances.shape[1]

print(f'number of classes: {len(train.classes_)}')
print(f'number of training documents: {len(train)}')
print(f'training prevalence: {F.strprev(train.prevalence())}')
print(f'training matrix shape: {train.instances.shape}')

dev_prev = pd.read_csv(os.path.join(path_binary_vector, 'public', 'dev_prevalences.csv'), index_col=0)
print(dev_prev)




scores = {}
for quantifier in [CC]: #, ACC, PCC, PACC, EMQ, HDy]:

    classifier = CalibratedClassifierCV(LogisticRegression())
    model = quantifier(classifier).fit(train)
    quantifier_name = model.__class__.__name__

    scores[quantifier_name]={}
    for sample_set, sample_size in [('dev', 1000)]:
        ae_errors, rae_errors = [], []
        for i, row in tqdm(dev_prev.iterrows(), total=len(dev_prev), desc=f'testing {quantifier_name} in {sample_set}'):
            filename = row['filename']
            prev_true = row[1:].values
            sample_path = os.path.join(path_binary_vector, 'public', f'{sample_set}_vectors', filename)
            sample, _ = load_binary_vectors(sample_path, nF)
            qp.environ['SAMPLE_SIZE'] = sample.shape[0]
            prev_estim = model.quantify(sample)
            # prev_true  = sample.prevalence()
            ae_errors.append(qp.error.mae(prev_true, prev_estim))
            rae_errors.append(qp.error.mrae(prev_true, prev_estim))

        ae_errors = np.asarray(ae_errors)
        rae_errors = np.asarray(rae_errors)

        mae = ae_errors.mean()
        mrae = rae_errors.mean()
        scores[quantifier_name][sample_set] = {'mae': mae, 'mrae': mrae}
        pickle.dump(ae_errors, open(os.path.join(result_path, f'{quantifier_name}.{sample_set}.ae.pickle'), 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(rae_errors, open(os.path.join(result_path, f'{quantifier_name}.{sample_set}.rae.pickle'), 'wb'), pickle.HIGHEST_PROTOCOL)
        print(f'{quantifier_name} {sample_set} MAE={mae:.4f}')
        print(f'{quantifier_name} {sample_set} MRAE={mrae:.4f}')

for model in scores:
    for sample_set in ['validation']:#, 'test']:
        print(f'{model}\t{scores[model][sample_set]["mae"]:.4f}\t{scores[model][sample_set]["mrae"]:.4f}')


"""
test:
CC	0.1859	1.5406
ACC	0.0453	0.2840
PCC	0.1793	1.7187
PACC	0.0287	0.1494
EMQ	0.0225	0.1020
HDy	0.0631	0.2307

validation
CC	0.1862	1.9587
ACC	0.0394	0.2669
PCC	0.1789	2.1383
PACC	0.0354	0.1587
EMQ	0.0224	0.0960
HDy	0.0467	0.2121
"""


