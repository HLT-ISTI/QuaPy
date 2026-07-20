"""
Integration tests for dataset fetchers and large external resources.

This module is intentionally excluded from default ``unittest`` discovery by
using an ``integration_*.py`` filename.
"""

import os
import unittest

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import quapy.functional as F
from quapy.data.datasets import *
from quapy.method.aggregative import PCC


class IntegrationDatasetsTest(unittest.TestCase):

    def new_quantifier(self):
        return PCC(LogisticRegression(C=0.001, max_iter=100))

    def _check_dataset(self, dataset):
        train, test = dataset.reduce().train_test
        q = self.new_quantifier()
        if len(train) > 500:
            train = train.sampling(500)
        q.fit(*dataset.training.Xy)
        estim_prevalences = q.predict(dataset.test.instances)
        self.assertTrue(F.check_prevalence_vector(estim_prevalences))

    def _check_samples(self, gen, q, max_samples_test=5, vectorizer=None):
        for X, p in gen():
            if vectorizer is not None:
                X = vectorizer.transform(X)
            estim_prevalences = q.predict(X)
            self.assertTrue(F.check_prevalence_vector(estim_prevalences))
            max_samples_test -= 1
            if max_samples_test == 0:
                break

    def test_reviews(self):
        for dataset_name in REVIEWS_SENTIMENT_DATASETS:
            dataset = fetch_reviews(dataset_name, tfidf=True, min_df=10)
            dataset.reduce()
            self._check_dataset(dataset)

    def test_twitter(self):
        for dataset_name in TWITTER_SENTIMENT_DATASETS_TEST[:1]:
            dataset = fetch_twitter(dataset_name, min_df=10)
            dataset.reduce()
            self._check_dataset(dataset)

    def test_UCIBinaryDataset(self):
        for dataset_name in UCI_BINARY_DATASETS:
            dataset = fetch_UCIBinaryDataset(dataset_name)
            dataset.reduce()
            self._check_dataset(dataset)

    def test_UCIMultiDataset(self):
        for dataset_name in UCI_MULTICLASS_DATASETS:
            dataset = fetch_UCIMulticlassDataset(dataset_name)
            n_classes = dataset.n_classes
            uniform_prev = F.uniform_prevalence(n_classes)
            dataset.training = dataset.training.sampling(100, *uniform_prev)
            dataset.test = dataset.test.sampling(100, *uniform_prev)
            self._check_dataset(dataset)

    def test_lequa2022(self):
        if os.environ.get('QUAPY_TESTS_OMIT_LARGE_DATASETS'):
            return

        for dataset_name in LEQUA2022_VECTOR_TASKS:
            train, gen_val, gen_test = fetch_lequa2022(dataset_name)
            n_classes = train.n_classes
            train = train.sampling(100, *F.uniform_prevalence(n_classes))
            q = self.new_quantifier()
            q.fit(*train.Xy)
            self._check_samples(gen_val, q, max_samples_test=5)
            self._check_samples(gen_test, q, max_samples_test=5)

        for dataset_name in LEQUA2022_TEXT_TASKS:
            train, gen_val, gen_test = fetch_lequa2022(dataset_name)
            n_classes = train.n_classes
            train = train.sampling(100, *F.uniform_prevalence(n_classes))
            tfidf = TfidfVectorizer()
            train.instances = tfidf.fit_transform(train.instances)
            q = self.new_quantifier()
            q.fit(*train.Xy)
            self._check_samples(gen_val, q, max_samples_test=5, vectorizer=tfidf)
            self._check_samples(gen_test, q, max_samples_test=5, vectorizer=tfidf)

    def test_lequa2024(self):
        if os.environ.get('QUAPY_TESTS_OMIT_LARGE_DATASETS'):
            return

        for task in LEQUA2024_TASKS:
            train, gen_val, gen_test = fetch_lequa2024(task, merge_T3=True)
            n_classes = train.n_classes
            train = train.sampling(100, *F.uniform_prevalence(n_classes))
            q = self.new_quantifier()
            q.fit(*train.Xy)
            self._check_samples(gen_val, q, max_samples_test=5)
            self._check_samples(gen_test, q, max_samples_test=5)

    def test_IFCB(self):
        if os.environ.get('QUAPY_TESTS_OMIT_LARGE_DATASETS'):
            return

        for mod_sel in [False, True]:
            train, gen = fetch_IFCB(single_sample_train=True, for_model_selection=mod_sel)
            n_classes = train.n_classes
            train = train.sampling(100, *F.uniform_prevalence(n_classes))
            q = self.new_quantifier()
            q.fit(*train.Xy)
            self._check_samples(gen, q, max_samples_test=5)


if __name__ == '__main__':
    unittest.main()
