from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import quapy as qp
from method.aggregative import PACC, CC, EMQ
from quapy.data import LabelledCollection
from os.path import join
from utils import load_samples
from evaluation import nmd

domain = 'Books'
datapath = './data'
protocol = 'app'

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2))

train = LabelledCollection.load(join(datapath, domain, 'training_data.txt'), loader_func=qp.data.reader.from_text)
train.instances = tfidf.fit_transform(train.instances)


def load_test_samples():
    for sample in load_samples(join(datapath, domain, protocol, 'test_samples'), classes=train.classes_):
        sample.instances = tfidf.transform(sample.instances)
        yield sample.instances, sample.prevalence()


q = EMQ(LogisticRegression())
q.fit(train)
report = qp.evaluation.gen_prevalence_report(q, gen_fn=load_test_samples, error_metrics=[nmd], eval_budget=100)
mean_nmd = report['nmd'].mean()
std_nmd = report['nmd'].std()

print(f'{mean_nmd:.4f} +-{std_nmd:.4f}')



