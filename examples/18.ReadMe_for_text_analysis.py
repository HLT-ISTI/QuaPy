from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

import quapy as qp
from quapy.method.non_aggregative import ReadMe
import quapy.functional as F
from sklearn.pipeline import Pipeline

"""
This example showcases how to use the non-aggregative method ReadMe proposed by Hopkins and King.
This method is for text analysis, so let us first instantiate a dataset for sentiment quantification (we
use IMDb for this example). The method is quite computationally expensive, so we will restrict the training
set to 1000 documents only.
"""
reviews = qp.datasets.fetch_reviews('imdb').reduce(n_train=1000, random_state=0)

"""
We need to convert text to bag-of-words representations. Actually, ReadMe requires the representations to be
binary (i.e., storing a 1 whenever a document contains certain word, or 0 otherwise), so we will not use
TFIDF weighting. We will also retain the top 1000 most important features according to chi2.
"""
encode_0_1 = Pipeline([
    ('0_1_terms', CountVectorizer(min_df=5, binary=True)),
    ('feat_sel',  SelectKBest(chi2, k=1000))
])
train, test = qp.data.preprocessing.instance_transformation(reviews, encode_0_1, inplace=True).train_test

"""
We now instantiate ReadMe, with the prob_model='full' (default behaviour, implementing the Hopkins and King original
idea). This method consists of estimating Q(Y) by solving:

Q(X) = \sum_i Q(X|Y=i) Q(Y=i)

without resorting to estimating the posteriors Q(Y=i|X), by solving a linear least-squares problem.
However, since Q(X) and Q(X|Y=i) are matrices of shape (2^K, 1) and (2^K, n), with K the number of features
and n the number of classes, their calculation becomes intractable. ReadMe instead performs bagging (i.e., it
samples small sets of features and averages the results) thus reducing K to a few terms. In our example we
set K (bagging_range) to 20, and the number of bagging_trials to 100. 

ReadMe also computes confidence intervals via bootstrap. We set the number of bootstrap trials to 100.  
"""
readme = ReadMe(prob_model='full', bootstrap_trials=100, bagging_trials=100, bagging_range=20, random_state=0, verbose=True)
readme.fit(*train.Xy)   # <- there is actually nothing happening here (only bootstrap resampling); the method is "lazy"
                        #    and postpones most of the calculations to the test phase.

# since the method is slow, we will only test 3 cases with different imbalances
few_negatives = [0.25, 0.75]
balanced = [0.5, 0.5]
few_positives = [0.75, 0.25]

for test_prev in [few_negatives, balanced, few_positives]:
    sample = reviews.test.sampling(500, *test_prev, random_state=0)  # draw sets of 500 documents with desired prevs
    prev_estim, conf = readme.predict_conf(sample.X)
    err = qp.error.mae(sample.prevalence(), prev_estim)
    print(f'true-prevalence={F.strprev(sample.prevalence())},\n'
          f'predicted-prevalence={F.strprev(prev_estim)}, with confidence intervals {conf},\n'
          f'MAE={err:.4f}')



