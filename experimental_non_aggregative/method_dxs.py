from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import quapy as qp
from data import LabelledCollection
import numpy as np

from experimental_non_aggregative.custom_vectorizers import *
from method._kdey import KDEBase
from protocol import APP
from quapy.method.aggregative import HDy, DistributionMatchingY
from quapy.method.base import BaseQuantifier
from scipy import optimize
import pandas as pd
import quapy.functional as F


# TODO: explore the bernoulli (term presence/absence) variant
# TODO: explore the multinomial (term frequency) variant
# TODO: explore the multinomial + length normalization variant
# TODO: consolidate the TSR-variant (e.g., using information gain) variant;
#   - works better with the idf?
#   - works better with length normalization?
#   - etc

class DxS(BaseQuantifier):
    def __init__(self, vectorizer=None, divergence='topsoe'):
        self.vectorizer = vectorizer
        self.divergence = divergence

    # def __as_distribution(self, instances):
    #     return np.asarray(instances.sum(axis=0) / instances.sum()).flatten()

    def __as_distribution(self, instances):
        dist = instances.mean(axis=0)
        return np.asarray(dist).flatten()

    def fit(self, text_instances, labels):

        classes = np.unique(labels)

        if self.vectorizer is not None:
            text_instances = self.vectorizer.fit_transform(text_instances, y=labels)

        distributions = []
        for class_i in classes:
            distributions.append(self.__as_distribution(text_instances[labels == class_i]))

        self.validation_distribution = np.asarray(distributions)

        return self

    def predict(self, text_instances):
        if self.vectorizer is not None:
            text_instances = self.vectorizer.transform(text_instances)

        test_distribution = self.__as_distribution(text_instances)
        divergence = qp.functional.get_divergence(self.divergence)
        n_classes, n_feats = self.validation_distribution.shape

        def match(prev):
            prev = np.expand_dims(prev, axis=0)
            mixture_distribution = (prev @ self.validation_distribution).flatten()
            return divergence(test_distribution, mixture_distribution)

        # the initial point is set as the uniform distribution
        uniform_distribution = np.full(fill_value=1 / n_classes, shape=(n_classes,))

        # solutions are bounded to those contained in the unit-simplex
        bounds = tuple((0, 1) for x in range(n_classes))  # values in [0,1]
        constraints = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})  # values summing up to 1
        r = optimize.minimize(match, x0=uniform_distribution, method='SLSQP', bounds=bounds, constraints=constraints)
        return r.x



class KDExML(BaseQuantifier, KDEBase):

    def __init__(self, bandwidth=0.1, standardize=False):
        self._check_bandwidth(bandwidth)
        self.bandwidth = bandwidth
        self.standardize = standardize

    def fit(self, X, y):
        classes = sorted(np.unique(y))

        if self.standardize:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        if issparse(X):
            X = X.toarray()

        self.mix_densities = self.get_mixture_components(X, y, classes, self.bandwidth)
        return self

    def predict(self, X):
        """
        Searches for the mixture model parameter (the sought prevalence values) that maximizes the likelihood
        of the data (i.e., that minimizes the negative log-likelihood)

        :param X: instances in the sample
        :return: a vector of class prevalence estimates
        """
        epsilon = 1e-10
        if issparse(X):
            X = X.toarray()
        n_classes = len(self.mix_densities)
        if self.standardize:
            X = self.scaler.transform(X)
        test_densities = [self.pdf(kde_i, X) for kde_i in self.mix_densities]

        def neg_loglikelihood(prev):
            test_mixture_likelihood = sum(prev_i * dens_i for prev_i, dens_i in zip (prev, test_densities))
            test_loglikelihood = np.log(test_mixture_likelihood + epsilon)
            return  -np.sum(test_loglikelihood)

        return F.optim_minimize(neg_loglikelihood, n_classes)



if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = 250
    qp.environ['N_JOBS'] = -1
    min_df = 10
    # dataset = 'imdb'
    repeats = 10
    error = 'mae'

    div = 'topsoe'

    # generates tuples (dataset, method, method_name)
    # (the dataset is needed for methods that process the dataset differently)
    def gen_methods():

        for dataset in qp.datasets.REVIEWS_SENTIMENT_DATASETS:

            data = qp.datasets.fetch_reviews(dataset, tfidf=False)

            # bernoulli_vectorizer = CountVectorizer(min_df=min_df, binary=True)
            # dxs = DxS(divergence=div, vectorizer=bernoulli_vectorizer)
            # yield data, dxs, 'DxS-Bernoulli'
            #
            # multinomial_vectorizer = CountVectorizer(min_df=min_df, binary=False)
            # dxs = DxS(divergence=div, vectorizer=multinomial_vectorizer)
            # yield data, dxs, 'DxS-multinomial'
            #
            # tf_vectorizer = TfidfVectorizer(sublinear_tf=False, use_idf=False, min_df=min_df, norm=None)
            # dxs = DxS(divergence=div, vectorizer=tf_vectorizer)
            # yield data, dxs, 'DxS-TF'
            #
            # logtf_vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=False, min_df=min_df, norm=None)
            # dxs = DxS(divergence=div, vectorizer=logtf_vectorizer)
            # yield data, dxs, 'DxS-logTF'
            #
            # tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=min_df, norm=None)
            # dxs = DxS(divergence=div, vectorizer=tfidf_vectorizer)
            # yield data, dxs, 'DxS-TFIDF'
            #
            # tfidf_vectorizer = TfidfVectorizer(use_idf=True, min_df=min_df, norm='l2')
            # dxs = DxS(divergence=div, vectorizer=tfidf_vectorizer)
            # yield data, dxs, 'DxS-TFIDF-l2'

            tsr_vectorizer = TSRweighting(tsr_function=information_gain, min_df=min_df, norm='l2')
            dxs = DxS(divergence=div, vectorizer=tsr_vectorizer)
            yield data, dxs, 'DxS-TFTSR-l2'

            data = qp.datasets.fetch_reviews(dataset, tfidf=True, min_df=min_df)

            kdex = KDExML()
            reduction = TruncatedSVD(n_components=100, random_state=0)
            red_data = qp.data.preprocessing.instance_transformation(data, transformer=reduction, inplace=False)
            yield red_data, kdex, 'KDEx'

            hdy = HDy(LogisticRegression())
            yield data, hdy, 'HDy'

            # dm = DistributionMatchingY(LogisticRegression(), divergence=div, nbins=5)
            # yield data, dm, 'DM-5b'
            #
            # dm = DistributionMatchingY(LogisticRegression(), divergence=div, nbins=10)
            # yield data, dm, 'DM-10b'




    result_path = 'results.csv'
    with open(result_path, 'wt') as csv:
        csv.write(f'Method\tDataset\tMAE\tMRAE\n')
        for data, quantifier, quant_name in gen_methods():
            quantifier.fit(*data.training.Xy)
            report = qp.evaluation.evaluation_report(quantifier, APP(data.test, repeats=repeats), error_metrics=['mae','mrae'], verbose=True)
            means = report.mean(numeric_only=True)
            csv.write(f'{quant_name}\t{data.name}\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\n')

    df = pd.read_csv(result_path, sep='\t')
    # print(df)

    pv = df.pivot_table(index='Method', columns="Dataset", values=["MAE", "MRAE"])
    print(pv)




