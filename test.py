from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import quapy as qp
import quapy.functional as F


SAMPLE_SIZE=500
binary = False
svmperf_home = './svm_perf_quantification'

if binary:
    # load a textual binary dataset and create a tfidf bag of words
    train_path = './datasets/reviews/kindle/train.txt'
    test_path = './datasets/reviews/kindle/test.txt'
    dataset = qp.Dataset.load(train_path, test_path, qp.reader.from_text)
    qp.preprocessing.text2tfidf(dataset, inplace=True)
    qp.preprocessing.reduce_columns(dataset, min_df=10, inplace=True)

else:
    # load a sparse matrix ternary dataset
    train_path = './datasets/twitter/train/sst.train+dev.feature.txt'
    test_path = './datasets/twitter/test/sst.test.feature.txt'
    dataset = qp.Dataset.load(train_path, test_path, qp.reader.from_sparse)
    dataset.training = dataset.training.sampling(SAMPLE_SIZE, 0.2, 0.5, 0.3)
    qp.preprocessing.reduce_columns(dataset, min_df=10, inplace=True)
    print(dataset.training.instances.shape)

print('dataset loaded')

# training a quantifier
learner = LogisticRegression()
# model = qp.method.aggregative.ClassifyAndCount(learner)
# model = qp.method.aggregative.AdjustedClassifyAndCount(learner)
# model = qp.method.aggregative.ProbabilisticClassifyAndCount(learner)
# model = qp.method.aggregative.ProbabilisticAdjustedClassifyAndCount(learner)
# model = qp.method.aggregative.ExpectationMaximizationQuantifier(learner)
# model = qp.method.aggregative.ExplicitLossMinimisationBinary(svmperf_home, loss='q', C=100)
model = qp.method.aggregative.SVMQ(svmperf_home, C=1)

if not binary:
    model = qp.method.aggregative.OneVsAll(model)

print('fitting model')
model.fit(dataset.training)


# estimating class prevalences
print('quantifying')
prevalences_estim = model.quantify(dataset.test.instances)
prevalences_true  = dataset.test.prevalence()

# evaluation (one single prediction)
error = qp.error.mae(prevalences_true, prevalences_estim)

print(f'method {model.__class__.__name__}')

print(f'Evaluation in test (1 eval)')
print(f'true prevalence {F.strprev(prevalences_true)}')
print(f'estim prevalence {F.strprev(prevalences_estim)}')
print(f'mae={error:.3f}')


max_evaluations = 5000
n_prevpoints = F.get_nprevpoints_approximation(combinations_budget=max_evaluations, n_classes=dataset.n_classes)
n_evaluations = F.num_prevalence_combinations(n_prevpoints, dataset.n_classes)
print(f'the prevalence interval [0,1] will be split in {n_prevpoints} prevalence points for each class, so that '
      f'the requested maximum number of sample evaluations ({max_evaluations}) is not exceeded. '
      f'For the {dataset.n_classes} classes this dataset has, this will yield a total of {n_evaluations} evaluations.')

true_prev, estim_prev = qp.evaluation.artificial_sampling_prediction(model, dataset.test, SAMPLE_SIZE, n_prevpoints)

qp.error.SAMPLE_SIZE = SAMPLE_SIZE
print(f'Evaluation according to the artificial sampling protocol ({len(true_prev)} evals)')
for error in qp.error.QUANTIFICATION_ERROR:
    score = error(true_prev, estim_prev)
    print(f'{error.__name__}={score:.5f}')

