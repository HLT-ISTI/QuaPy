from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import quapy as qp
import quapy.functional as F
import sys
import numpy as np

#qp.datasets.fetch_reviews('hp')
#qp.datasets.fetch_twitter('sst')

#sys.exit()
from model_selection import GridSearchQ

SAMPLE_SIZE=500
binary = False
svmperf_home = './svm_perf_quantification'

if binary:
    dataset = qp.datasets.fetch_reviews('kindle', tfidf=True, min_df=5)

else:
    dataset = qp.datasets.fetch_twitter('hcr', for_model_selection=False, min_df=10, pickle=True)
    # dataset.training = dataset.training.sampling(SAMPLE_SIZE, 0.2, 0.5, 0.3)

print('dataset loaded')

# training a quantifier
learner = LogisticRegression(max_iter=1000)
# model = qp.method.aggregative.ClassifyAndCount(learner)
model = qp.method.aggregative.AdjustedClassifyAndCount(learner)
# model = qp.method.aggregative.ProbabilisticClassifyAndCount(learner)
# model = qp.method.aggregative.ProbabilisticAdjustedClassifyAndCount(learner)
# model = qp.method.aggregative.ExpectationMaximizationQuantifier(learner)
# model = qp.method.aggregative.ExplicitLossMinimisationBinary(svmperf_home, loss='q', C=100)
# model = qp.method.aggregative.SVMQ(svmperf_home, C=1)

if not binary and isinstance(model, qp.method.aggregative.BinaryQuantifier):
    model = qp.method.aggregative.OneVsAll(model)


# Model fit and Evaluation on the test data
# ----------------------------------------------------------------------------

print(f'fitting model {model.__class__.__name__}')
train, val = dataset.training.split_stratified(0.6)
model.fit(train, val_split=val)

# estimating class prevalences
print('quantifying')
prevalences_estim = model.quantify(dataset.test.instances)
prevalences_true  = dataset.test.prevalence()

# evaluation (one single prediction)
error = qp.error.mae(prevalences_true, prevalences_estim)

print(f'Evaluation in test (1 eval)')
print(f'true prevalence {F.strprev(prevalences_true)}')
print(f'estim prevalence {F.strprev(prevalences_estim)}')
print(f'mae={error:.3f}')


# Model fit and Evaluation according to the artificial sampling protocol
# ----------------------------------------------------------------------------

max_evaluations = 5000
n_prevpoints = F.get_nprevpoints_approximation(combinations_budget=max_evaluations, n_classes=dataset.n_classes)
n_evaluations = F.num_prevalence_combinations(n_prevpoints, dataset.n_classes)
print(f'the prevalence interval [0,1] will be split in {n_prevpoints} prevalence points for each class, so that\n'
      f'the requested maximum number of sample evaluations ({max_evaluations}) is not exceeded.\n'
      f'For the {dataset.n_classes} classes this dataset has, this will yield a total of {n_evaluations} evaluations.')

true_prev, estim_prev = qp.evaluation.artificial_sampling_prediction(model, dataset.test, SAMPLE_SIZE, n_prevpoints)

qp.error.SAMPLE_SIZE = SAMPLE_SIZE
print(f'Evaluation according to the artificial sampling protocol ({len(true_prev)} evals)')
for error in qp.error.QUANTIFICATION_ERROR:
    score = error(true_prev, estim_prev)
    print(f'{error.__name__}={score:.5f}')


# Model selection and Evaluation according to the artificial sampling protocol
# ----------------------------------------------------------------------------

param_grid = {'C': np.logspace(-3,3,7), 'class_weight': ['balanced', None]}

model_selection = GridSearchQ(model,
                              param_grid=param_grid,
                              sample_size=SAMPLE_SIZE,
                              eval_budget=max_evaluations//10,
                              error='mae',
                              refit=True,
                              verbose=True)

# model = model_selection.fit(dataset.training, validation=0.3)
model = model_selection.fit(train, validation=val)
print(f'Model selection: best_params = {model_selection.best_params_}')
print(f'param scores:')
for params, score in model_selection.param_scores_.items():
    print(f'\t{params}: {score:.5f}')

true_prev, estim_prev = qp.evaluation.artificial_sampling_prediction(model, dataset.test, SAMPLE_SIZE, n_prevpoints)

print(f'After model selection: Evaluation according to the artificial sampling protocol ({len(true_prev)} evals)')
for error in qp.error.QUANTIFICATION_ERROR:
    score = error(true_prev, estim_prev)
    print(f'{error.__name__}={score:.5f}')