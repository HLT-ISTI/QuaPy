from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import quapy as qp
import quapy.functional as F
from method.aggregative import OneVsAll

# load a textual binary dataset and create a tfidf bag of words
#from method.aggregative import OneVsAll, BaseQuantifier

train_path = './datasets/reviews/kindle/train.txt'
test_path = './datasets/reviews/kindle/test.txt'
#dataset = qp.Dataset.load(train_path, test_path, qp.reader.from_text)
#dataset.training = dataset.training.sampling(1000, 0.4, 0.6)
#dataset.test = dataset.test.sampling(500, 0.6, 0.4)
#qp.preprocessing.text2tfidf(dataset, inplace=True)
#qp.preprocessing.reduce_columns(dataset, min_df=10, inplace=True)

# load a sparse matrix ternary dataset
train_path = './datasets/twitter/train/sst.train+dev.feature.txt'
test_path = './datasets/twitter/test/sst.test.feature.txt'
dataset = qp.Dataset.load(train_path, test_path, qp.reader.from_sparse)
dataset.training = dataset.training.sampling(500, 0.3, 0.4, 0.3)
dataset.test = dataset.test.sampling(500, 0.2, 0.5, 0.3)

# training a quantifier
learner = LogisticRegression()
# q = qp.method.aggregative.ClassifyAndCount(learner)
# q = qp.method.aggregative.AdjustedClassifyAndCount(learner)
# q = qp.method.aggregative.AdjustedClassifyAndCount(learner)
# q = qp.method.aggregative.ProbabilisticClassifyAndCount(learner)
# q = qp.method.aggregative.ProbabilisticAdjustedClassifyAndCount(learner)
# q = qp.method.aggregative.ExpectationMaximizationQuantifier(learner)
# q = qp.method.aggregative.ExplicitLossMinimisation(svmperf_base='./svm_perf_quantification', loss='q', verbose=0, C=1000)
# q = qp.method.aggregative.SVMQ(svmperf_base='./svm_perf_quantification', verbose=0, C=1000)
#model = qp.method.aggregative.HDy(learner)
#

model = qp.method.aggregative.HDy(learner)
model = OneVsAll(model)
print(model.get_params())

model.fit(dataset.training)

# estimating class prevalences
prevalences_estim = model.quantify(dataset.test.instances)
prevalences_true  = dataset.test.prevalence()

# evaluation (one single prediction)
error = qp.error.mae(prevalences_true, prevalences_estim)

print(f'method {model.__class__.__name__}')
print(f'true prevalence {F.strprev(prevalences_true)}')
print(f'estim prevalence {F.strprev(prevalences_estim)}')
print(f'MAE={error:.3f}')