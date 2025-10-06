import os
import unittest

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import quapy.functional as F
from quapy.method.aggregative import PCC
from quapy.data.datasets import *


class TestDatasets(unittest.TestCase):

    def new_quantifier(self):
        return PCC(LogisticRegression(C=0.001, max_iter=100))

    def _check_dataset(self, dataset):
        q = self.new_quantifier()
        print(f'testing method {q} in {dataset.name}...', end='')
        q.fit(dataset.training)
        estim_prevalences = q.quantify(dataset.test.instances)
        self.assertTrue(F.check_prevalence_vector(estim_prevalences))
        print(f'[done]')

    def _check_samples(self, gen, q, max_samples_test=5, vectorizer=None):
        for X, p in gen():
            if vectorizer is not None:
                X = vectorizer.transform(X)
            estim_prevalences = q.quantify(X)
            self.assertTrue(F.check_prevalence_vector(estim_prevalences))
            max_samples_test -= 1
            if max_samples_test == 0:
                break

    def test_reviews(self):
        for dataset_name in REVIEWS_SENTIMENT_DATASETS:
            print(f'loading dataset {dataset_name}...', end='')
            dataset = fetch_reviews(dataset_name, tfidf=True, min_df=10)
            dataset.stats()
            dataset.reduce()
            print(f'[done]')
            self._check_dataset(dataset)

    def test_twitter(self):
        for dataset_name in TWITTER_SENTIMENT_DATASETS_TEST:
            print(f'loading dataset {dataset_name}...', end='')
            dataset = fetch_twitter(dataset_name, min_df=10)
            dataset.stats()
            dataset.reduce()
            print(f'[done]')
            self._check_dataset(dataset)

    def test_UCIBinaryDataset(self):
        for dataset_name in UCI_BINARY_DATASETS:
            try:
                print(f'loading dataset {dataset_name}...', end='')
                dataset = fetch_UCIBinaryDataset(dataset_name)
                dataset.stats()
                dataset.reduce()
                print(f'[done]')
                self._check_dataset(dataset)
            except FileNotFoundError as fnfe:
                if dataset_name == 'pageblocks.5' and fnfe.args[0].find(
                        'If this is the first time you attempt to load this dataset') > 0:
                    print('The pageblocks.5 dataset requires some hand processing to be usable; skipping this test.')
                    continue

    def test_UCIMultiDataset(self):
        for dataset_name in UCI_MULTICLASS_DATASETS:
            print(f'loading dataset {dataset_name}...', end='')
            dataset = fetch_UCIMulticlassDataset(dataset_name)
            dataset.stats()
            n_classes = dataset.n_classes
            uniform_prev = F.uniform_prevalence(n_classes)
            dataset.training = dataset.training.sampling(100, *uniform_prev)
            dataset.test = dataset.test.sampling(100, *uniform_prev)
            print(f'[done]')
            self._check_dataset(dataset)

    def test_lequa2022(self):
        if os.environ.get('QUAPY_TESTS_OMIT_LARGE_DATASETS'):
            print("omitting test_lequa2022 because QUAPY_TESTS_OMIT_LARGE_DATASETS is set")
            return

        for dataset_name in LEQUA2022_VECTOR_TASKS:
            print(f'loading dataset {dataset_name}...', end='')
            train, gen_val, gen_test = fetch_lequa2022(dataset_name)
            train.stats()
            n_classes = train.n_classes
            train = train.sampling(100, *F.uniform_prevalence(n_classes))
            q = self.new_quantifier()
            q.fit(train)
            self._check_samples(gen_val, q, max_samples_test=5)
            self._check_samples(gen_test, q, max_samples_test=5)

        for dataset_name in LEQUA2022_TEXT_TASKS:
            print(f'loading dataset {dataset_name}...', end='')
            train, gen_val, gen_test = fetch_lequa2022(dataset_name)
            train.stats()
            n_classes = train.n_classes
            train = train.sampling(100, *F.uniform_prevalence(n_classes))
            tfidf = TfidfVectorizer()
            train.instances = tfidf.fit_transform(train.instances)
            q = self.new_quantifier()
            q.fit(train)
            self._check_samples(gen_val, q, max_samples_test=5, vectorizer=tfidf)
            self._check_samples(gen_test, q, max_samples_test=5, vectorizer=tfidf)


    def test_IFCB(self):
        if os.environ.get('QUAPY_TESTS_OMIT_LARGE_DATASETS'):
            print("omitting test_IFCB because QUAPY_TESTS_OMIT_LARGE_DATASETS is set")
            return

        print(f'loading dataset IFCB.')
        for mod_sel in [False, True]:
            train, gen = fetch_IFCB(single_sample_train=True, for_model_selection=mod_sel)
            train.stats()
            n_classes = train.n_classes
            train = train.sampling(100, *F.uniform_prevalence(n_classes))
            q = self.new_quantifier()
            q.fit(train)
            self._check_samples(gen, q, max_samples_test=5)


if __name__ == '__main__':
    unittest.main()
