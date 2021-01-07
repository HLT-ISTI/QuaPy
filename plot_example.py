from sklearn.model_selection import GridSearchCV
import numpy as np
import quapy as qp
from sklearn.linear_model import LogisticRegression

sample_size = 500
qp.environ['SAMPLE_SIZE'] = sample_size



def gen_data():

    data = qp.datasets.fetch_reviews('kindle', tfidf=True, min_df=5)

    models = [
        qp.method.aggregative.CC,
        qp.method.aggregative.ACC,
        qp.method.aggregative.PCC,
        qp.method.aggregative.PACC,
        qp.method.aggregative.HDy,
        qp.method.aggregative.EMQ,
        qp.method.meta.ECC,
        qp.method.meta.EACC,
        qp.method.meta.EHDy,
    ]

    method_names, true_prevs, estim_prevs, tr_prevs = [], [], [], []
    for Quantifier in models:
        print(f'training {Quantifier.__name__}')
        lr = LogisticRegression(max_iter=1000, class_weight='balanced')
        # lr = GridSearchCV(lr, param_grid={'C':np.logspace(-3,3,7)}, n_jobs=-1)
        model = Quantifier(lr).fit(data.training)
        true_prev, estim_prev = qp.evaluation.artificial_sampling_prediction(
            model, data.test, sample_size, n_repetitions=20, n_prevpoints=11)

        method_names.append(Quantifier.__name__)
        true_prevs.append(true_prev)
        estim_prevs.append(estim_prev)
        tr_prevs.append(data.training.prevalence())

    return method_names, true_prevs, estim_prevs, tr_prevs

method_names, true_prevs, estim_prevs, tr_prevs = qp.util.pickled_resource('./plots/plot_data.pkl', gen_data)

qp.plot.error_by_drift(method_names, true_prevs, estim_prevs, tr_prevs, n_bins=11, savepath='./plots/err_drift.png')
qp.plot.binary_diagonal(method_names, true_prevs, estim_prevs, savepath='./plots/bin_diag.png')
qp.plot.binary_bias_global(method_names, true_prevs, estim_prevs, savepath='./plots/bin_bias.png')
qp.plot.binary_bias_bins(method_names, true_prevs, estim_prevs, nbins=11, savepath='./plots/bin_bias_bin.png')
