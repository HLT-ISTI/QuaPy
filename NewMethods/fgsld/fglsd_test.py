from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from fgsld.fgsld_quantifiers import FakeFGLSD
from method.aggregative import EMQ, CC
import quapy as qp


qp.environ['SAMPLE_SIZE'] = 500

dataset = qp.datasets.fetch_reviews('kindle')
qp.data.preprocessing.text2tfidf(dataset, min_df=5, inplace=True)

training = dataset.training
test = dataset.test

cls = CalibratedClassifierCV(LinearSVC())


method_names, true_prevs, estim_prevs, tr_prevs = [], [], [], []

for model, model_name in [
    (CC(cls), 'CC'),
    # (FakeFGLSD(cls, nbins=5, isomerous=False, recompute_bins=False), 'FGSLD-isometric-stat-5'),
    (FakeFGLSD(cls, nbins=5, isomerous=True, recompute_bins=True), 'FGSLD-isometric-dyn-5'),
    # (FakeFGLSD(cls, nbins=5, isomerous=True, recompute_bins=False), 'FGSLD-isomerous-stat-5'),
    # (FakeFGLSD(cls, nbins=10, isomerous=True, recompute_bins=True), 'FGSLD-isomerous-dyn-10'),
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
        model, test, qp.environ['SAMPLE_SIZE'], n_repetitions=5, n_prevpoints=11, n_jobs=-1
    )
    method_names.append(model_name)
    true_prevs.append(true_prev)
    estim_prevs.append(estim_prev)
    tr_prevs.append(training.prevalence())


qp.plot.binary_diagonal(method_names, true_prevs, estim_prevs, train_prev=tr_prevs[0], savepath='./plot_fglsd.png')
