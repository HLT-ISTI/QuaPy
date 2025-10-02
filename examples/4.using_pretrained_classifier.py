"""
Aggregative quantifiers use an underlying classifier. Often, one has one pre-trained classifier available, and
needs to use this classifier at the basis of a quantification system. In such cases, the classifier should not
be retrained, but only used to issue classifier predictions for the quantifier.
In this example, we show how to instantiate a quantifier with a pre-trained classifier.
"""
from typing import List, Dict

import quapy as qp
from quapy.method.aggregative import PACC
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import pipeline
import numpy as np
import quapy.functional as F


# A scikit-learn's style wrapper for a huggingface-based pre-trained transformer
class HFTextClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
        self.pipe = pipeline("sentiment-analysis", model=model_name)
        self.classes_ = np.asarray([0,1])

    def fit(self, X, y=None):
        return self

    def _binary_decisions(self, transformer_output: List[Dict]):
        return np.array([(1 if p['label']=='POSITIVE' else 0) for p in transformer_output], dtype=int)

    def predict(self, X):
        X = list(map(str, X))
        preds = self.pipe(X, truncation=True)
        return self._binary_decisions(preds)

    def predict_proba(self, X):
        X = list(map(str, X))
        n_examples = len(X)
        preds = self.pipe(X, truncation=True)
        decisions = self._binary_decisions(preds)
        scores = np.array([p['score'] for p in preds], dtype=float)
        probas = np.zeros(shape=(len(X), 2), dtype=float)
        probas[np.arange(n_examples),decisions] = scores
        probas[np.arange(n_examples),~decisions] = 1-scores
        return probas

# load a sentiment dataset
dataset = qp.datasets.fetch_reviews('imdb', tfidf=False)  # raw text
train, test = dataset.training, dataset.test

# instantiate a pre-trained classifier
clf = HFTextClassifier()

# Let us fit a quantifier based on our pre-trained classifier.
# Note that, since the classifier is already fit, we will use the entire training set for
# learning the aggregation function of the quantifier.
# To do so, we only need to indicate "fit_classifier"=False, as follows:
quantifier = PACC(clf, fit_classifier=False)   # Probabilistic Classify & Count using a pre-trained model

print('training PACC...')
quantifier.fit(*train.Xy)

# let us simulate some shifted test data...
new_prevalence = [0.75, 0.25]
shifted_test = test.sampling(500, *new_prevalence, random_state=0)

# and do some evaluation
print('predicting with PACC...')
estim_prevalence = quantifier.predict(shifted_test.X)

print('Result:\n'+('='*20))
print(f'training prevalence: {F.strprev(train.prevalence())}')
print(f'(shifted) test prevalence: {F.strprev(shifted_test.prevalence())}')
print(f'estimated prevalence: {F.strprev(estim_prevalence)}')

absolute_error = qp.error.ae(new_prevalence, estim_prevalence)
print(f'absolute error={absolute_error:.4f}')