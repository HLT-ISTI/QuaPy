import pickle
from sklearn.linear_model import LogisticRegression
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

param_grid = {
    'C': np.logspace(-3,3,7),
    'class_weight': ['balanced', None]
}


def gen_samples():
    return gen_load_samples_T1(T1A_devvectors_path, nF, ground_truth_path=T1A_devprevalence_path, return_id=False)


for quantifier in [EMQ]: # [CC, ACC, PCC, PACC, EMQ, HDy]:
    if quantifier == EMQ:
        classifier = CalibratedClassifierCV(LogisticRegression(), n_jobs=-1)
    else:
        classifier = LogisticRegression()
    model = quantifier(classifier)
    print(f'{model.__class__.__name__}: Model selection')
    model = qp.model_selection.GridSearchQ(
        model,
        param_grid,
        sample_size=None,
        protocol='gen',
        error=qp.error.mae,
        refit=False,
        verbose=True
    ).fit(train, gen_samples)

    quantifier_name = model.best_model().__class__.__name__
    print(f'{quantifier_name} mae={model.best_score_:.3f} (params: {model.best_params_})')

    pickle.dump(model.best_model(),
                open(os.path.join(models_path, quantifier_name+'.pkl'), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)


"""
validation
CC	0.1862	1.9587
ACC	0.0394	0.2669
PCC	0.1789	2.1383
PACC	0.0354	0.1587
EMQ	0.0224	0.0960
HDy	0.0467	0.2121
"""


