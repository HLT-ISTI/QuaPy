import quapy as qp
from quapy.method.aggregative import MS2, OneVsAllAggregative, OneVsAllGeneric, SVMQ
from quapy.method.base import getOneVsAll
from quapy.model_selection import GridSearchQ
from quapy.protocol import USimplexPP
from sklearn.linear_model import LogisticRegression
import numpy as np

"""
In this example, we will create a quantifier for tweet sentiment analysis considering three classes: negative, neutral,
and positive. We will use a one-vs-all approach using a binary quantifier for demonstration purposes.
"""

qp.environ['SAMPLE_SIZE'] = 100
qp.environ['N_JOBS'] = -1
qp.environ['SVMPERF_HOME'] = '../svm_perf_quantification'

"""
Any binary quantifier can be turned into a single-label quantifier by means of getOneVsAll function.
This function returns an instance of OneVsAll quantifier. Actually, it either returns the subclass OneVsAllGeneric 
when the quantifier is an instance of BaseQuantifier, and it returns OneVsAllAggregative when the quantifier is
an instance of AggregativeQuantifier. Although OneVsAllGeneric works in all cases, using OneVsAllAggregative has 
some additional advantages (namely, all the advantages that AggregativeQuantifiers enjoy, i.e., faster predictions
during evaluation).
"""
quantifier = getOneVsAll(SVMQ())
print(f'the quantifier is an instance of {quantifier.__class__.__name__}')

# load a ternary dataset
train_modsel, val = qp.datasets.fetch_twitter('hcr', for_model_selection=True, pickle=True).train_test

"""
model selection: for this example, we are relying on the USimplexPP protocol, i.e., a variant of the 
artificial-prevalence protocol that generates random samples (100 in this case) for randomly picked priors 
from the unit simplex. The priors are sampled using the Kraemer algorithm. Note this is in contrast to the 
standard APP protocol, that instead explores a prefixed grid of prevalence values.
"""
param_grid = {
    'binary_quantifier__classifier__C': np.logspace(-2,2,5),  # classifier-dependent hyperparameter
}
print('starting model selection')
model_selection = GridSearchQ(quantifier, param_grid, protocol=USimplexPP(val), verbose=True, refit=False)
quantifier = model_selection.fit(train_modsel).best_model()

print('training on the whole training set')
train, test = qp.datasets.fetch_twitter('hcr', for_model_selection=False, pickle=True).train_test
quantifier.fit(train)

# evaluation
mae = qp.evaluation.evaluate(quantifier, protocol=USimplexPP(test), error_metric='mae')

print(f'MAE = {mae:.4f}')


