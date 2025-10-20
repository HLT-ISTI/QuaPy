from sklearn.feature_extraction.text import CountVectorizer
import quapy as qp
from quapy.method.non_aggregative import ReadMe
import quapy.functional as F

reviews = qp.datasets.fetch_reviews('imdb').reduce(n_train=1000, random_state=0)

encode_0_1 = CountVectorizer(min_df=5, binary=True)
train, test = qp.data.preprocessing.instance_transformation(reviews, encode_0_1, inplace=True).train_test

readme = ReadMe(bootstrap_trials=100, bagging_trials=100, bagging_range=100, random_state=0, verbose=True)
readme.fit(*train.Xy)

for test_prev in [[0.25, 0.75], [0.5, 0.5], [0.75, 0.25]]:
    sample = reviews.test.sampling(500, *test_prev, random_state=0)
    prev_estim, conf = readme.predict_conf(sample.X)
    err = qp.error.mae(sample.prevalence(), prev_estim)
    print(f'true-prevalence={F.strprev(sample.prevalence())},\n'
          f'predicted-prevalence={F.strprev(prev_estim)},\n'
          f'MAE={err:.4f}')
    print(conf)


