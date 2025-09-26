import quapy as qp
from quapy.method.aggregative import newELM
from quapy.method.base import newOneVsAll
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP

"""
In this example, we will show hoy to define a quantifier based on explicit loss minimization (ELM).
ELM is a family of quantification methods relying on structured output learning. In particular, we will
showcase how to instantiate SVM(Q) as proposed by `Barranquero et al. 2015 
<https://www.sciencedirect.com/science/article/pii/S003132031400291X>`_, and SVM(KLD) and SVM(nKLD) as proposed by
`Esuli et al. 2015 <https://dl.acm.org/doi/abs/10.1145/2700406>`_.

All ELM quantifiers rely on SVMperf for optimizing a structured loss function (Q, KLD, or nKLD). Since these are
not part of the original SVMperf package by Joachims,  you have to first download the SVMperf package, apply the 
patch svm-perf-quantification-ext.patch (provided with QuaPy library), and compile the sources. 
The script prepare_svmperf.sh does all the job. Simply run:

>>> ./prepare_svmperf.sh

Note that ELM quantifiers are nothing but a classify and count (CC) model instantiated with SVMperf as the 
underlying classifier. E.g., SVM(Q) comes down to:

>>> CC(SVMperf(svmperf_base, loss='q'))

this means that ELM are aggregative quantifiers (since CC is an aggregative quantifier). QuaPy provides some helper 
functions for simplify this; for example:

>>> newSVMQ(svmperf_base)

returns an instance of SVM(Q) (i.e., an instance of CC properly set to work with SVMperf optimizing for Q.

Since we wan to explore the losses, we will instead use newELM. For this example we will create a quantifier for tweet 
sentiment analysis considering three classes: negative, neutral, and positive. Since SVMperf is a binary classifier, 
our quantifier will be binary as well. We will use a one-vs-all approach to work in multiclass model. 
For more details about how one-vs-all works, we refer to the example "10.one_vs_all.py" and to the API documentation. 
"""

qp.environ['SAMPLE_SIZE'] = 100
qp.environ['N_JOBS'] = -1
qp.environ['SVMPERF_HOME'] = '../svm_perf_quantification'

quantifier = newOneVsAll(newELM())
print(f'the quantifier is an instance of {quantifier.__class__.__name__}')

# load a ternary dataset
train_modsel, val = qp.datasets.fetch_twitter('hcr', for_model_selection=True, pickle=True).train_test

"""
model selection: 
We explore the classifier's loss and the classifier's C hyperparameters.
Since our model is actually an instance of OneVsAllAggregative, we need to add the prefix "binary_quantifier", and
since our binary quantifier is an instance of CC (an aggregative quantifier), we need to add the prefix "classifier".
"""
param_grid = {
    'binary_quantifier__classifier__loss': ['q', 'kld', 'mae'],  # classifier-dependent hyperparameter
    'binary_quantifier__classifier__C': [0.01, 1, 100],  # classifier-dependent hyperparameter
}
print('starting model selection')
model_selection = GridSearchQ(quantifier, param_grid, protocol=UPP(val), verbose=True, refit=False)
quantifier = model_selection.fit(*train_modsel.Xy).best_model()

print('training on the whole training set')
train, test = qp.datasets.fetch_twitter('hcr', for_model_selection=False, pickle=True).train_test
quantifier.fit(*train.Xy)

# evaluation
mae = qp.evaluation.evaluate(quantifier, protocol=UPP(test), error_metric='mae')

print(f'MAE = {mae:.4f}')


