from typing import Union, Callable
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.neighbors import KernelDensity

import quapy as qp
from data import LabelledCollection
from protocol import APP, UPP
from quapy.method.aggregative import AggregativeProbabilisticQuantifier, _training_helper, cross_generate_predictions, \
    DistributionMatching, _get_divergence
import scipy
from scipy import optimize


# TODO: optimize the bandwidth automatically
# TODO: replace the l2 metric in the kernel with the EMD, try to visualize the difference between both criteria in a 3-simplex
# TODO: think of a MMD-y variant, i.e., a MMD variant that uses the points in the simplex and possibly any non-linear kernel


class KDEy(AggregativeProbabilisticQuantifier):

    BANDWIDTH_METHOD = ['auto', 'scott', 'silverman']
    ENGINE = ['scipy', 'sklearn']

    def __init__(self, classifier: BaseEstimator, val_split=0.4, divergence: Union[str, Callable]='HD',
                 bandwidth_method='scott', engine='sklearn', n_jobs=None):
        self.classifier = classifier
        self.val_split = val_split
        self.divergence = divergence
        self.bandwidth_method = bandwidth_method
        self.engine = engine
        self.n_jobs = n_jobs
        assert bandwidth_method in KDEy.BANDWIDTH_METHOD, f'unknown bandwidth_method, valid ones are {KDEy.BANDWIDTH_METHOD}'
        assert engine in KDEy.ENGINE, f'unknown engine, valid ones are {KDEy.ENGINE}'

    def get_kde(self, posteriors):
        if self.engine == 'scipy':
            # scipy treats columns as datapoints, and need the datapoints not to lie in a lower-dimensional subspace, which
            # requires removing the last dimension which is constrained
            posteriors = posteriors[:,:-1].T
            kde = scipy.stats.gaussian_kde(posteriors)
            kde.set_bandwidth(self.bandwidth_method)
        elif self.engine == 'sklearn':
            kde = KernelDensity(bandwidth=self.bandwidth_method).fit(posteriors)
        return kde

    def pdf(self, kde, posteriors):
        if self.engine == 'scipy':
            return kde(posteriors[:,:-1].T)
        elif self.engine == 'sklearn':
            return np.exp(kde.score_samples(posteriors))

    def fit(self, data: LabelledCollection, fit_classifier=True, val_split: Union[float, LabelledCollection] = None):
        """
        Trains the classifier (if requested) and generates the validation distributions out of the training data.
        The validation distributions have shape `(n, ch, nbins)`, with `n` the number of classes, `ch` the number of
        channels (a channel is a description, in form of a histogram, of a specific class -- there are as many channels
        as classes, although in the binary case one can use only one channel, since the other one is constrained),
        and `nbins` the number of bins. In particular, let `V` be the validation distributions; `di=V[i]`
        are the distributions obtained from training data labelled with class `i`; `dij = di[j]` is the discrete
        distribution of posterior probabilities `P(Y=j|X=x)` for training data labelled with class `i`, and `dij[k]`
        is the fraction of instances with a value in the `k`-th bin.

        :param data: the training set
        :param fit_classifier: set to False to bypass the training (the learner is assumed to be already fit)
        :param val_split: either a float in (0,1) indicating the proportion of training instances to use for
         validation (e.g., 0.3 for using 30% of the training set as validation data), or a LabelledCollection
         indicating the validation set itself, or an int indicating the number k of folds to be used in kFCV
         to estimate the parameters
        """
        if val_split is None:
            val_split = self.val_split

        self.classifier, y, posteriors, classes, class_count = cross_generate_predictions(
            data, self.classifier, val_split, probabilistic=True, fit_classifier=fit_classifier, n_jobs=self.n_jobs
        )

        self.val_densities = [self.get_kde(posteriors[y == cat]) for cat in range(data.n_classes)]
        self.val_posteriors = posteriors

        return self

    def val_pdf(self, prev):
        """
        Returns a function that computes the mixture model with the given prev as mixture factor
        :param prev: a prevalence vector, ndarray
        :return: a function implementing the validation distribution with fixed mixture factor
        """
        return lambda posteriors: sum(prev_i * self.pdf(kde_i, posteriors) for kde_i, prev_i in zip(self.val_densities, prev))


    def aggregate(self, posteriors: np.ndarray):
        """
        Searches for the mixture model parameter (the sought prevalence values) that yields a validation distribution
        (the mixture) that best matches the test distribution, in terms of the divergence measure of choice.
        In the multiclass case, with `n` the number of classes, the test and mixture distributions contain
        `n` channels (proper distributions of binned posterior probabilities), on which the divergence is computed
        independently. The matching is computed as an average of the divergence across all channels.

        :param instances: instances in the sample
        :return: a vector of class prevalence estimates
        """
        test_density = self.get_kde(posteriors)
        # val_test_posteriors = np.concatenate([self.val_posteriors, posteriors])
        test_likelihood = self.pdf(test_density, posteriors)
        divergence = _get_divergence(self.divergence)


        n_classes = len(self.val_densities)

        def match(prev):
            val_pdf = self.val_pdf(prev)
            val_likelihood  = val_pdf(posteriors)
            return divergence(val_likelihood, test_likelihood)

        # the initial point is set as the uniform distribution
        uniform_distribution = np.full(fill_value=1 / n_classes, shape=(n_classes,))

        # solutions are bounded to those contained in the unit-simplex
        bounds = tuple((0, 1) for _ in range(n_classes))  # values in [0,1]
        constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})  # values summing up to 1
        r = optimize.minimize(match, x0=uniform_distribution, method='SLSQP', bounds=bounds, constraints=constraints)
        return r.x



if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = 100
    qp.environ['N_JOBS'] = -1
    div = 'HD'

    # generates tuples (dataset, method, method_name)
    # (the dataset is needed for methods that process the dataset differently)
    def gen_methods():

        for dataset in qp.datasets.TWITTER_SENTIMENT_DATASETS_TEST:

            data = qp.datasets.fetch_twitter(dataset, min_df=3, pickle=True)

            # kdey = KDEy(LogisticRegression(), divergence=div, bandwidth_method='scott')
            # yield data, kdey, f'KDEy-{div}-scott'

            kdey = KDEy(LogisticRegression(), divergence=div, bandwidth_method='silverman', engine='sklearn')
            yield data, kdey, f'KDEy-{div}-silverman'

            dm = DistributionMatching(LogisticRegression(), divergence=div, nbins=5)
            yield data, dm, f'DM-5b-{div}'

            # dm = DistributionMatching(LogisticRegression(), divergence=div, nbins=10)
            # yield data, dm, f'DM-10b-{div}'


    result_path = 'results_kdey.csv'
    with open(result_path, 'wt') as csv:
        csv.write(f'Method\tDataset\tMAE\tMRAE\n')
        for data, quantifier, quant_name in gen_methods():
            quantifier.fit(data.training)
            protocol = UPP(data.test, repeats=100)
            report = qp.evaluation.evaluation_report(quantifier, protocol, error_metrics=['mae','mrae'], verbose=True)
            means = report.mean()
            csv.write(f'{quant_name}\t{data.name}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\n')
            csv.flush()

    df = pd.read_csv(result_path, sep='\t')
    # print(df)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pv = df.pivot_table(index='Dataset', columns="Method", values=["MAE", "MRAE"])
    print(pv)
