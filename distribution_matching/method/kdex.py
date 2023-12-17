from quapy.method.base import BaseQuantifier
import numpy as np
from distribution_matching.method.kdey import KDEBase

import quapy as qp
from quapy.data import LabelledCollection
import quapy.functional as F
from sklearn.preprocessing import StandardScaler


class KDExML(BaseQuantifier, KDEBase):

    def __init__(self, bandwidth=0.1, standardize=True):
        self._check_bandwidth(bandwidth)
        self.bandwidth = bandwidth
        self.standardize = standardize

    def fit(self, data: LabelledCollection):
        X, y = data.Xy
        if self.standardize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        self.mix_densities = self.get_mixture_components(X, y, data.n_classes, self.bandwidth)
        return self

    def quantify(self, X):
        """
        Searches for the mixture model parameter (the sought prevalence values) that maximizes the likelihood
        of the data (i.e., that minimizes the negative log-likelihood)

        :param X: instances in the sample 
        :return: a vector of class prevalence estimates
        """
        epsilon = 1e-10
        n_classes = len(self.mix_densities)
        if self.standardize:
            X = self.scaler.transform(X)
        test_densities = [self.pdf(kde_i, X) for kde_i in self.mix_densities]

        def neg_loglikelihood(prev):
            test_mixture_likelihood = sum(prev_i * dens_i for prev_i, dens_i in zip (prev, test_densities))
            test_loglikelihood = np.log(test_mixture_likelihood + epsilon)
            return  -np.sum(test_loglikelihood)

        return F.optim_minimize(neg_loglikelihood, n_classes)




