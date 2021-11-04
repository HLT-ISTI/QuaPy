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

predictions_path = os.path.join('predictions', 'T1B')  # multiclass - vector
os.makedirs(predictions_path, exist_ok=True)

pathT1B = './data/T1B/public'
T1B_devvectors_path = os.path.join(pathT1B, 'dev_vectors')
T1B_devprevalence_path = os.path.join(pathT1B, 'dev_prevalences.csv')
T1B_trainpath = os.path.join(pathT1B, 'training_vectors.txt')
T1B_catmap = os.path.join(pathT1B, 'training_vectors_label_map.txt')

train = LabelledCollection.load(T1B_trainpath, load_binary_vectors)
nF = train.instances.shape[1]

qp.environ['SAMPLE_SIZE'] = constants.T1B_SAMPLE_SIZE

print(f'number of classes: {len(train.classes_)}')
print(f'number of training documents: {len(train)}')
print(f'training prevalence: {F.strprev(train.prevalence())}')
print(f'training matrix shape: {train.instances.shape}')

true_prevalence = ResultSubmission.load(T1B_devprevalence_path)

cat2code, categories = load_category_map(T1B_catmap)

for quantifier in [PACC]:  # [CC, ACC, PCC, PACC, EMQ]:

    classifier = CalibratedClassifierCV(LogisticRegression())
    model = quantifier(classifier).fit(train)
    quantifier_name = model.__class__.__name__

    predictions = ResultSubmission(categories=categories)
    for samplename, sample in tqdm(gen_load_samples_T1(T1B_devvectors_path, nF),
                                   desc=quantifier_name, total=len(true_prevalence)):
        predictions.add(samplename, model.quantify(sample))

    predictions.dump(os.path.join(predictions_path, quantifier_name + '.csv'))
    mae, mrae = evaluate_submission(true_prevalence, predictions)
    print(f'{quantifier_name} mae={mae:.3f} mrae={mrae:.3f}')



