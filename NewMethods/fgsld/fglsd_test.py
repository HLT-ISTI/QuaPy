from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from NewMethods.fgsld.fine_grained_sld import FineGrainedSLD
from method.aggregative import EMQ, CC
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier
import quapy as qp
import quapy.functional as F
from sklearn.linear_model import LogisticRegression


class FakeFGLSD(BaseQuantifier):
    def __init__(self, learner, nbins, isomerous):
        self.learner = learner
        self.nbins = nbins
        self.isomerous = isomerous

    def fit(self, data: LabelledCollection):
        self.Xtr, self.ytr = data.Xy
        self.learner.fit(self.Xtr, self.ytr)
        return self

    def quantify(self, instances):
        tr_priors = F.prevalence_from_labels(self.ytr, n_classes=2)
        fgsld = FineGrainedSLD(self.Xtr, instances, self.ytr, tr_priors, self.learner, n_bins=self.nbins)
        priors, posteriors = fgsld.run(self.isomerous)
        return priors

    def get_params(self, deep=True):
        pass

    def set_params(self, **parameters):
        pass



qp.environ['SAMPLE_SIZE'] = 500

dataset = qp.datasets.fetch_reviews('hp')
qp.data.preprocessing.text2tfidf(dataset, min_df=5, inplace=True)

training = dataset.training
test = dataset.test

cls = CalibratedClassifierCV(LinearSVC())


method_names, true_prevs, estim_prevs, tr_prevs = [], [], [], []

for model, model_name in [
    (CC(cls), 'CC'),
    (FakeFGLSD(cls, nbins=1, isomerous=False), 'FGSLD-1'),
    (FakeFGLSD(cls, nbins=2, isomerous=False), 'FGSLD-2'),
    #(FakeFGLSD(cls, nbins=5, isomerous=False), 'FGSLD-5'),
    #(FakeFGLSD(cls, nbins=10, isomerous=False), 'FGSLD-10'),
    #(FakeFGLSD(cls, nbins=50, isomerous=False), 'FGSLD-50'),
    #(FakeFGLSD(cls, nbins=100, isomerous=False), 'FGSLD-100'),
#    (FakeFGLSD(cls, nbins=1, isomerous=False), 'FGSLD-1'),
    #(FakeFGLSD(cls, nbins=10, isomerous=True), 'FGSLD-10-ISO'),
    # (FakeFGLSD(cls, nbins=50, isomerous=False), 'FGSLD-50'),
    (EMQ(cls), 'SLD'),
]:
    print('running ', model_name)
    model.fit(training)
    true_prev, estim_prev = qp.evaluation.artificial_sampling_prediction(
        model, test, qp.environ['SAMPLE_SIZE'], n_repetitions=10, n_prevpoints=21, n_jobs=-1
    )
    method_names.append(model_name)
    true_prevs.append(true_prev)
    estim_prevs.append(estim_prev)
    tr_prevs.append(training.prevalence())


qp.plot.binary_diagonal(method_names, true_prevs, estim_prevs, train_prev=tr_prevs[0], savepath='./plot_fglsd.png')
