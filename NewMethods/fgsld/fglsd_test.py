from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from fgsld_quantifiers import FakeFGLSD
from method.aggregative import EMQ, CC
import quapy as qp
import numpy as np


qp.environ['SAMPLE_SIZE'] = 500

dataset = qp.datasets.fetch_reviews('hp')
qp.data.preprocessing.text2tfidf(dataset, min_df=5, inplace=True)

training = dataset.training
test = dataset.test

cls = CalibratedClassifierCV(LinearSVC())

#cls = LogisticRegression()


method_names, true_prevs, estim_prevs, tr_prevs = [], [], [], []

for model, model_name in [
    (CC(cls), 'CC'),
#    (FakeFGLSD(cls, nbins=20, isomerous=False, recompute_bins=True), 'FGSLD-isometric-dyn-20'),
    #(FakeFGLSD(cls, nbins=11, isomerous=False, recompute_bins=True), 'FGSLD-isometric-dyn-11'),
    #(FakeFGLSD(cls, nbins=8, isomerous=False, recompute_bins=True), 'FGSLD-isometric-dyn-8'),
    #(FakeFGLSD(cls, nbins=6, isomerous=False, recompute_bins=True), 'FGSLD-isometric-dyn-6'),
    (FakeFGLSD(cls, nbins=5, isomerous=False, recompute_bins=True), 'FGSLD-isometric-dyn-5'),
    #(FakeFGLSD(cls, nbins=4, isomerous=False, recompute_bins=True), 'FGSLD-isometric-dyn-4'),
    #(FakeFGLSD(cls, nbins=3, isomerous=False, recompute_bins=True), 'FGSLD-isometric-dyn-3'),
    (FakeFGLSD(cls, nbins=1, isomerous=False, recompute_bins=True), 'FGSLD-isometric-dyn-1'),
#    (FakeFGLSD(cls, nbins=3, isomerous=False, recompute_bins=False), 'FGSLD-isometric-sta-3'),
    (EMQ(cls), 'SLD'),
]:
    print('running ', model_name)
    model.fit(training)
    true_prev, estim_prev = qp.evaluation.artificial_sampling_prediction(
        model, test, qp.environ['SAMPLE_SIZE'], n_repetitions=5, n_prevpoints=11, n_jobs=-1
    )
    method_names.append(model_name)
    true_prevs.append(true_prev)
    estim_prevs.append(estim_prev)
    tr_prevs.append(training.prevalence())
    #if hasattr(model, 'iterations'):
    #    print(f'iterations ave={np.mean(model.iterations):.3f}, min={np.min(model.iterations):.3f}, max={np.max(model.iterations):.3f}')


qp.plot.binary_diagonal(method_names, true_prevs, estim_prevs, train_prev=tr_prevs[0], savepath='./plot_fglsd.png')
