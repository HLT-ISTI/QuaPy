import numpy as np
from sklearn.linear_model import LogisticRegression

import quapy as qp
from quapy.method.aggregative import PACC
from quapy.data import LabelledCollection
from quapy.protocol import AbstractStochasticSeededProtocol
import quapy.functional as F

"""
In this example, we create a custom protocol.
The protocol generates samples of a Gaussian mixture model with random mixture parameter (the sample prevalence).
Datapoints are univariate and we consider 2 classes only.
"""
class GaussianMixProtocol(AbstractStochasticSeededProtocol):
    # We need to extend AbstractStochasticSeededProtocol if we want the samples to be replicable

    def __init__(self, mu_1:float, std_1:float, mu_2:float, std_2:float, num_samples, sample_size, random_state=0):
        super(GaussianMixProtocol, self).__init__(random_state)  # this sets the random state
        self.mu_1 = mu_1
        self.std_1 = std_1
        self.mu_2 = mu_2
        self.std_2 = std_2
        self.num_samples = num_samples
        self.sample_size = sample_size

    def samples_parameters(self):
        # This function is inherited and has to be overriden.
        # This function should return all the necessary parameters for producing the samples.
        # In this case, we consider returning a vector of seeds (one for each sample) and a vector of
        # randomly sampled prevalence values.
        # This function will be invoked within a context that sets the seed, so it will always return the
        # same parameters. In case you want different outcomes, then simply set random_state=None.
        rand_offset = np.random.randint(1000)
        sample_seeds = np.random.permutation(self.num_samples*2) + rand_offset
        random_prevs = np.random.rand(self.num_samples)
        params = np.hstack([sample_seeds.reshape(-1,2), random_prevs.reshape(-1,1)])
        # each row in params contains two seeds (for generating the negatives and the positives, respectively) and
        # the prevalence vector
        return params

    def sample(self, params):
        # the params are two seeds and the positive prevalence of the sample
        seed0, seed1, pos_prev = params
        num_positives = int(pos_prev * self.sample_size)
        num_negatives = self.sample_size - num_positives
        with qp.util.temp_seed(int(seed0)):
            Xneg = np.random.normal(loc=self.mu_1, scale=self.std_1, size=num_negatives)
        with qp.util.temp_seed(int(seed1)):
            Xpos = np.random.normal(loc=self.mu_2, scale=self.std_2, size=num_positives)
        X = np.concatenate((Xneg,Xpos))
        np.random.shuffle(X)
        X = X.reshape(-1,1)
        prev = F.as_binary_prevalence(pos_prev)
        return X, prev

    def total(self):
        # overriding this function will allow some methods display a meaningful progress bar
        return self.num_samples


mu_1, std_1 = 0, 1
mu_2, std_2 = 1, 1

gm = GaussianMixProtocol(mu_1=mu_1, std_1=std_1, mu_2=mu_2, std_2=std_2, num_samples=10, sample_size=50)

# let's see if the samples are replicated
for i, (X, prev) in enumerate(gm()):
    if i>4: break
    print(f'sample-{i}: {F.strprev(prev)}, some covariates={X[:5].flatten()}...')

print()
for i, (X, prev) in enumerate(gm()):
    if i > 4: break
    print(f'sample-{i}: {F.strprev(prev)}, some covariates={X[:5].flatten()}...')

# let's generate some training data
# The samples are replicable, but by setting a temp seed we achieve repicable training as well
with qp.util.temp_seed(0):
    Xneg = np.random.normal(loc=mu_1, scale=std_1, size=100)
    Xpos = np.random.normal(loc=mu_2, scale=std_2, size=100)
    X = np.concatenate([Xneg, Xpos]).reshape(-1,1)
    y = [0]*100 + [1]*100
    training = LabelledCollection(X, y)

    pacc = PACC(LogisticRegression())
    pacc.fit(training)


mae = qp.evaluation.evaluate(pacc, protocol=gm, error_metric='mae', verbose=True)
print(f'PACC MAE={mae:.5f}')


