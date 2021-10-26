import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pandas as pd

import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import *
import quapy.functional as F
from data import *
import os
import constants


# LeQua official baselines for task T1A (Binary/Vector)
# =====================================================

predictions_path = os.path.join('predictions', 'T1A')
os.makedirs(predictions_path, exist_ok=True)

models_path = os.path.join('models', 'T1A')
os.makedirs(models_path, exist_ok=True)

pathT1A = './data/T1A/public'
T1A_devvectors_path = os.path.join(pathT1A, 'dev_vectors')
T1A_devprevalence_path = os.path.join(pathT1A, 'dev_prevalences.csv')
T1A_trainpath = os.path.join(pathT1A, 'training_vectors.txt')

train = LabelledCollection.load(T1A_trainpath, load_binary_vectors)
nF = train.instances.shape[1]

qp.environ['SAMPLE_SIZE'] = constants.T1A_SAMPLE_SIZE

print(f'number of classes: {len(train.classes_)}')
print(f'number of training documents: {len(train)}')
print(f'training prevalence: {F.strprev(train.prevalence())}')
print(f'training matrix shape: {train.instances.shape}')

true_prevalence = ResultSubmission.load(T1A_devprevalence_path)

for quantifier in [CC, ACC, PCC, PACC, EMQ, HDy]:

    # classifier = CalibratedClassifierCV(LogisticRegression(C=1))
    classifier = LogisticRegression(C=1)
    model = quantifier(classifier).fit(train)
    quantifier_name = model.__class__.__name__

    predictions = ResultSubmission(categories=['negative', 'positive'])
    for samplename, sample in tqdm(gen_load_samples_T1(T1A_devvectors_path, nF),
                                   desc=quantifier_name, total=len(true_prevalence)):
        predictions.add(samplename, model.quantify(sample))

    predictions.dump(os.path.join(predictions_path, quantifier_name + '.csv'))
    pickle.dump(model, open(os.path.join(models_path, quantifier_name+'.pkl'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    mae, mrae = evaluate_submission(true_prevalence, predictions)
    print(f'{quantifier_name} mae={mae:.3f} mrae={mrae:.3f}')

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


