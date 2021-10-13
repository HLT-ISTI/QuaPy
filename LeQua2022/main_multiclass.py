import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import *
from data import load_multiclass_raw_document
import os

path_multiclass_raw = 'multiclass_raw'
result_path = os.path.join('results', 'multiclass_raw')
os.makedirs(result_path, exist_ok=True)

train_file = os.path.join(path_multiclass_raw, 'documents', 'training.txt')

train = LabelledCollection.load(train_file, load_multiclass_raw_document)

print('classes', train.classes_)
print('#classes', len(train.classes_))
print('#docs', len(train))
print('prevalence', train.prevalence())
print('counts', train.counts())

tfidf = TfidfVectorizer(min_df=5)
train.instances = tfidf.fit_transform(train.instances)
print(train.instances.shape[1])

scores = {}
for quantifier in [CC, ACC, PCC, PACC, EMQ]:#, HDy]:
    classifier = CalibratedClassifierCV(LogisticRegression())
    # classifier = LogisticRegression()
    model = quantifier(classifier).fit(train)
    print('model trained')

    quantifier_name = model.__class__.__name__
    scores[quantifier_name]={}
    for sample_set, sample_size in [('validation', 1000), ('test', 5000)]:
        ae_errors, rae_errors = [], []
        for i in tqdm(range(sample_size), total=sample_size, desc=f'testing {quantifier_name} in {sample_set}'):
            test_file = os.path.join(path_multiclass_raw, 'documents', f'{sample_set}_{i}.txt')
            test = LabelledCollection.load(test_file, load_multiclass_raw_document, classes=train.classes_)
            test.instances = tfidf.transform(test.instances)
            qp.environ['SAMPLE_SIZE'] = len(test)
            prev_estim = model.quantify(test.instances)
            prev_true  = test.prevalence()
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
    for sample_set in ['validation', 'test']:
        print(f'{model}\t{sample_set}\t{scores[model][sample_set]["mae"]:.4f}\t{scores[model][sample_set]["mrae"]:.4f}')


"""
test:


validation

"""


