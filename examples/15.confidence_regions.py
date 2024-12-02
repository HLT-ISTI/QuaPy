from quapy.method.confidence import AggregativeBootstrap
from quapy.method.aggregative import PACC
import quapy.functional as F
import quapy as qp

"""
Just like any other type of estimator, quantifier predictions are affected by error. It is therefore useful to provide,
along with the point estimate (the class prevalence values) a measure of uncertainty. These, typically come in the 
form of credible regions around the point estimate. 

QuaPy implements a method for deriving confidence regions around point estimates of class prevalence based on bootstrap.

Bootstrap method comes down to resampling the population several times, thus generating a series of point estimates.
QuaPy provides a variant of bootstrap for aggregative quantifiers, that only applies resampling to the pre-classified
instances.

Let see one example:  
"""

# load some data
data = qp.datasets.fetch_UCIMulticlassDataset('molecular')
train, test = data.train_test

# by simply wrapping an aggregative quantifier within the AggregativeBootstrap class, we can obtain confidence
# intervals around the point estimate, in this case, at 95% of confidence
pacc = AggregativeBootstrap(PACC(), n_test_samples=500, confidence_level=0.95)

with qp.util.temp_seed(0):
    # we train the quantifier the usual way
    pacc.fit(train)

    # let us simulate some shift in the test data
    random_prevalence = F.uniform_prevalence_sampling(n_classes=test.n_classes)
    shifted_test = test.sampling(200, *random_prevalence)
    true_prev = shifted_test.prevalence()

    # by calling "quantify_conf", we obtain the point estimate and the confidence intervals around it
    pred_prev, conf_intervals = pacc.quantify_conf(shifted_test.X)

    # conf_intervals is an instance of ConfidenceRegionABC, which provides some useful utilities like:
    # - coverage: a function which computes the fraction of true values that belong to the confidence region
    # - simplex_proportion: estimates the proportion of the simplex covered by the confidence region (amplitude)
    # ideally, we are interested in obtaining confidence regions with high level of coverage and small amplitude

    # the point estimate is computed as the mean of all bootstrap predictions; let us see the prediction error
    error = qp.error.ae(true_prev, pred_prev)

    # some useful outputs
    print(f'train prevalence: {F.strprev(train.prevalence())}')
    print(f'test prevalence:  {F.strprev(true_prev)}')
    print(f'point-estimate:   {F.strprev(pred_prev)}')
    print(f'absolute error:   {error:.3f}')
    print(f'Is the true value in the confidence region?: {conf_intervals.coverage(true_prev)==1}')
    print(f'Proportion of simplex covered at {pacc.confidence_level*100:.1f}%: {conf_intervals.simplex_portion()*100:.2f}%')

"""
Final remarks: 
There are various ways for performing bootstrap:
- the population-based approach (default): performs resampling of the test instances
    e.g., use  AggregativeBootstrap(PACC(), n_train_samples=1, n_test_samples=100, confidence_level=0.95)
- the model-based approach: performs resampling of the training instances, thus training several quantifiers
    e.g., use  AggregativeBootstrap(PACC(), n_train_samples=100, n_test_samples=1, confidence_level=0.95)
    this implementation avoids retraining the classifier, and performs resampling only to train different aggregation functions 
- the combined approach: a combination of the above
    e.g., use  AggregativeBootstrap(PACC(), n_train_samples=100, n_test_samples=100, confidence_level=0.95)
    this example will generate 100 x 100 predictions
    
There are different ways for constructing confidence regions implemented in QuaPy:
- confidence intervals: the simplest way, and one that typically works well in practice
    use: AggregativeBootstrap(PACC(), confidence_level=0.95, method='intervals')
- confidence ellipse in the simplex: creates an ellipse, which lies on the probability simplex, around the point estimate
    use: AggregativeBootstrap(PACC(), confidence_level=0.95, method='ellipse')
- confidence ellipse in the Centered-Log Ratio (CLR) space: creates an ellipse in the CLR space (this should be 
    convenient for taking into account the inner structure of the probability simplex)
    use: AggregativeBootstrap(PACC(), confidence_level=0.95, method='ellipse-clr')
    
Other methods that return confidence regions in QuaPy include the BayesianCC method.
"""


