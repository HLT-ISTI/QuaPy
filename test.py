from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, LinearSVR
import quapy as qp
import quapy.functional as F
import sys
import numpy as np

from NewMethods.methods import AveragePoolQuantification
from classification.methods import PCALR
from classification.neural import NeuralClassifierTrainer, CNNnet
from method.meta import EPACC
from quapy.model_selection import GridSearchQ

dataset = qp.datasets.fetch_UCIDataset('transfusion', verbose=True)
sys.exit(0)


qp.environ['SAMPLE_SIZE'] = 500
#param_grid = {'C': np.logspace(-3,3,7), 'class_weight': ['balanced', None]}
param_grid = {'C': np.logspace(0,3,4), 'class_weight': ['balanced']}
max_evaluations = 500

sample_size = qp.environ['SAMPLE_SIZE']
binary = False
svmperf_home = './svm_perf_quantification'

if binary:
    #dataset = qp.datasets.fetch_reviews('kindle', tfidf=True, min_df=5)
    dataset = qp.datasets.fetch_UCIDataset('german', verbose=True)
    #qp.data.preprocessing.index(dataset, inplace=True)

else:
    dataset = qp.datasets.fetch_twitter('gasp', for_model_selection=True, min_df=5, pickle=True)
    #dataset.training = dataset.training.sampling(sample_size, 0.2, 0.5, 0.3)

print(f'dataset loaded: #training={len(dataset.training)} #test={len(dataset.test)}')


# training a quantifier
# learner = LogisticRegression(max_iter=1000)
#model = qp.method.aggregative.ClassifyAndCount(learner)
# model = qp.method.aggregative.AdjustedClassifyAndCount(learner)
# model = qp.method.aggregative.ProbabilisticClassifyAndCount(learner)
# model = qp.method.aggregative.ProbabilisticAdjustedClassifyAndCount(learner)
# model = qp.method.aggregative.HellingerDistanceY(learner)
# model = qp.method.aggregative.ExpectationMaximizationQuantifier(learner)
# model = qp.method.aggregative.ExplicitLossMinimisationBinary(svmperf_home, loss='q', C=100)
# model = qp.method.aggregative.SVMQ(svmperf_home, C=1)

#learner = PCALR()
#learner = NeuralClassifierTrainer(CNNnet(dataset.vocabulary_size, dataset.n_classes))
#print(learner.get_params())
#model = qp.method.meta.QuaNet(learner, sample_size, device='cpu')

#learner = GridSearchCV(LogisticRegression(max_iter=1000), param_grid=param_grid, n_jobs=-1, verbose=1)
#learner = LogisticRegression(max_iter=1000)
# model = qp.method.aggregative.ClassifyAndCount(learner)

param_mod_sel = {
    'sample_size': 100,
    'n_prevpoints': 21,
    'n_repetitions': 5,
    'verbose': False
}
common = {
    'max_sample_size': 50,
    'n_jobs': -1,
    'param_grid': {'C': np.logspace(0,2,2), 'class_weight': ['balanced']},
    'param_mod_sel': param_mod_sel,
    'val_split': 0.4,
    'min_pos': 10,
    'size':6,
    'red_size':3
}

# hyperparameters will be evaluated within each quantifier of the ensemble, and so the typical model selection
# will be skipped (by setting hyperparameters to None)
model = EPACC(LogisticRegression(max_iter=100), optim='mrae', policy='mrae', **common)

"""    
Problemas:
- La interfaz es muy fea, hay que conocer practicamente todos los detalles así que no ahorra nada con respecto a crear
    un objeto con otros anidados dentro
- El fit genera las prevalences random, y esto hace que despues de la model selection, un nuevo fit tire todo el trabajo
    hecho.
- El fit de un GridSearcQ tiene dentro un best_estimator, pero después de la model selection, hacer fit otra vez sobre
    este objeto no se limita a re-entrenar el modelo con los mejores parámetros, sino que inicia una nueva búsqueda 
    en modo grid search.
- Posible solución (no vale): sería hacer directamente model selection con el benchmark final, aunque esto haría que los hyper-
    parámetros se buscasen en un conjunto diferente del resto de models....
- Posible solución: 
    - Elegir las prevalences en init
    - 
- Problema: el parámetro val_split es muy ambiguo en todo el framework. Por ejemplo, en EPACC podría ser un float que,
    en el caso de un GridSearchQ podría referir al split de validación para los hyperparámetros o al split que usa PACC
    para encontrar los parámetros...
"""

# regressor = LinearSVR(max_iter=10000)
# param_grid = {'C': np.logspace(-1,3,5)}
# model = AveragePoolQuantification(regressor, sample_size, trials=5000, n_components=500, zscore=False)

# model = qp.method.meta.EHDy(learner, param_grid=param_grid, optim='mae',
#                           sample_size=sample_size, eval_budget=max_evaluations//10, n_jobs=-1)
#model = qp.method.aggregative.ClassifyAndCount(learner)

# model = qp.method.meta.QuaNet(PCALR(n_components=100, max_iter=1000),
#                                sample_size=100,
#                                patience=10,
#                                tr_iter_per_poch=500, va_iter_per_poch=100, #lstm_nlayers=2, lstm_hidden_size=64,
#                                ff_layers=[500, 250, 50],
                               # checkpointdir='./checkpoint', device='cuda')

if qp.isbinary(model) and not qp.isbinary(dataset):
    model = qp.method.aggregative.OneVsAll(model)


# Model fit and Evaluation on the test data
# ----------------------------------------------------------------------------

print(f'fitting model {model.__class__.__name__}')
#train, val = dataset.training.split_stratified(0.6)
#model.fit(train, val_split=val)
qp.SAMPLE=1
qp.environ['SAMPLE_SIZE']=2
model.fit(dataset.training)





# estimating class prevalences
# print('quantifying')
# prevalences_estim = model.quantify(dataset.test.instances)
# prevalences_true  = dataset.test.prevalence()
#
# evaluation (one single prediction)
# error = qp.error.mae(prevalences_true, prevalences_estim)
#
# print(f'Evaluation in test (1 eval)')
# print(f'true prevalence {F.strprev(prevalences_true)}')
# print(f'estim prevalence {F.strprev(prevalences_estim)}')
# print(f'mae={error:.3f}')


# Model fit and Evaluation according to the artificial sampling protocol
# ----------------------------------------------------------------------------

n_prevpoints = F.get_nprevpoints_approximation(combinations_budget=max_evaluations, n_classes=dataset.n_classes)
n_evaluations = F.num_prevalence_combinations(n_prevpoints, dataset.n_classes)
print(f'the prevalence interval [0,1] will be split in {n_prevpoints} prevalence points for each class, so that\n'
      f'the requested maximum number of sample evaluations ({max_evaluations}) is not exceeded.\n'
      f'For the {dataset.n_classes} classes this dataset has, this will yield a total of {n_evaluations} evaluations.')

true_prev, estim_prev = qp.evaluation.artificial_sampling_prediction(model, dataset.test, sample_size, n_prevpoints)

#qp.error.SAMPLE_SIZE = sample_size
print(f'Evaluation according to the artificial sampling protocol ({len(true_prev)} evals)')
for error in qp.error.QUANTIFICATION_ERROR:
    score = error(true_prev, estim_prev)
    print(f'{error.__name__}={score:.5f}')

sys.exit(0)
# Model selection and Evaluation according to the artificial sampling protocol
# ----------------------------------------------------------------------------

model_selection = GridSearchQ(model,
                              param_grid=param_grid,
                              sample_size=sample_size,
                              eval_budget=max_evaluations//10,
                              error='mae',
                              refit=True,
                              verbose=True,
                              timeout=60*60)

model = model_selection.fit(dataset.training, val_split=0.3)
#model = model_selection.fit(train, validation=val)
print(f'Model selection: best_params = {model_selection.best_params_}')
print(f'param scores:')
for params, score in model_selection.param_scores_.items():
    print(f'\t{params}: {score:.5f}')

true_prev, estim_prev = qp.evaluation.artificial_sampling_prediction(model, dataset.test, sample_size, n_prevpoints)

print(f'After model selection: Evaluation according to the artificial sampling protocol ({len(true_prev)} evals)')
for error in qp.error.QUANTIFICATION_ERROR:
    score = error(true_prev, estim_prev)
    print(f'{error.__name__}={score:.5f}')