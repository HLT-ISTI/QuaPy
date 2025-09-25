import quapy as qp
import numpy as np

from protocol import APP
from quapy.method.aggregative import CC, ACC, PCC, PACC
from sklearn.svm import LinearSVC

qp.environ['SAMPLE_SIZE'] = 500


'''
In this example, we show how to create some plots for the analysis of experimental results.
The main functions are included in qp.plot but, before, we will generate some basic experimental data
'''

def gen_data():
    # this function generates some experimental data to plot

    def base_classifier():
        return LinearSVC(class_weight='balanced')

    def datasets():
        # the plots can handle experiments in different datasets
        yield qp.datasets.fetch_reviews('kindle', tfidf=True, min_df=5).train_test
        # by uncommenting thins line, the experiments will be carried out in more than one dataset
        # yield qp.datasets.fetch_reviews('hp', tfidf=True, min_df=5).train_test

    def models():
        yield 'CC', CC(base_classifier())
        yield 'ACC', ACC(base_classifier())
        yield 'PCC', PCC(base_classifier())
        yield 'PACC', PACC(base_classifier())

    # these are the main parameters we need to fill for generating the plots;
    # note that each these list must have the same number of elements, since the ith entry of each list regards
    # an independent experiment
    method_names, true_prevs, estim_prevs, tr_prevs = [], [], [], []

    for train, test in datasets():
        for method_name, model in models():
            model.fit(*train.Xy)
            true_prev, estim_prev = qp.evaluation.prediction(model, APP(test, repeats=100, random_state=0))

            # gather all the data for this experiment
            method_names.append(method_name)
            true_prevs.append(true_prev)
            estim_prevs.append(estim_prev)
            tr_prevs.append(train.prevalence())

    return method_names, true_prevs, estim_prevs, tr_prevs

# generate some experimental data
method_names, true_prevs, estim_prevs, tr_prevs = gen_data()
# if you want to play around with the different plots and parameters, you might prefer to generate the data only once,
# so you better replace the above line of code with this one, that pickles the experimental results for faster reuse
# method_names, true_prevs, estim_prevs, tr_prevs = qp.util.pickled_resource('./plots/data.pickle', gen_data)

# if there is only one training prevalence, we can display it
only_train_prev = tr_prevs[0] if len(np.unique(tr_prevs, axis=0))==1 else None

# diagonal plot (useful for analyzing the performance of quantifiers on binary data)
qp.plot.binary_diagonal(method_names, true_prevs, estim_prevs,
                        train_prev=only_train_prev, savepath='./plots/bin_diag.png')

# bias plot (box plots displaying the bias of each method)
qp.plot.binary_bias_global(method_names, true_prevs, estim_prevs, savepath='./plots/bin_bias.png')

# error by drift allows to plot the quantification error as a function of the amount of prior probability shift, and
# is preferable than diagonal plots for multiclass datasets
qp.plot.error_by_drift(method_names, true_prevs, estim_prevs, tr_prevs,
                       error_name='ae', n_bins=10, savepath='./plots/err_drift.png')

# each functions return (fig, ax) objects from matplotlib; use them to customize the plots to your liking
