import numpy as np
from NewMethods.fgsld.metrics import isomerous_bins, isometric_bins
from NewMethods.fgsld.em import History, get_measures_single_history
from sklearn.model_selection import cross_val_predict
import math
from scipy.special import softmax

class FineGrainedSLD:
    def __init__(self, x_tr, x_te, y_tr, tr_priors, clf, n_bins=10):
        self.y_tr = y_tr
        self.clf = clf
        self.tr_priors = tr_priors
        self.te_preds = clf.predict_proba(x_te)
        self.tr_preds = cross_val_predict(clf, x_tr, y_tr, method='predict_proba', n_jobs=10)
        self.n_bins = n_bins
        self.history: [History] = []
        self.multi_class = False

    def run(self, isomerous_binning, epsilon=1e-6, compute_bins_at_every_iter=True):
        """
        Run the FGSLD algorithm.

        :param isomerous_binning: whether to use isomerous or isometric binning.
        :param epsilon: stopping condition.
        :param compute_bins_at_every_iter: whether FGSLD should recompute the posterior bins at every iteration or not.
        :param return_posteriors_hist: whether to return posteriors at every iteration or not.
        :return: If `return_posteriors_hist` is true, the returned posteriors will be a list of numpy arrays, else a single numpy array with posteriors at last iteration.
        """
        smoothing_tr = 1e-9  # 1 / (2 * self.tr_preds.shape[0])
        smoothing_te = 1e-9  # 1 / (2 * self.te_preds.shape[0])
        s = 0
        tr_bin_priors = np.zeros((self.n_bins, self.tr_preds.shape[1]), dtype=np.float)
        te_bin_priors = np.zeros((self.n_bins, self.te_preds.shape[1]), dtype=np.float)
        tr_bins = self.__create_bins(training=True, isomerous_binning=isomerous_binning)
        self.__compute_bins_priors(tr_bin_priors, self.tr_preds, tr_bins, smoothing_tr)

        te_preds_cp = self.te_preds.copy()
        val = 2 * epsilon
        while not val < epsilon and s < 1000:
            if compute_bins_at_every_iter or s==0:
                te_bins = self.__create_bins(training=False, isomerous_binning=isomerous_binning)

            if s == 0:
                te_bin_priors_prev = tr_bin_priors.copy()
            else:
                te_bin_priors_prev = te_bin_priors.copy()
            self.__compute_bins_priors(te_bin_priors, self.te_preds, te_bins, smoothing_te)

            for label_idx, bins in te_bins.items():
                for i, bin_ in enumerate(bins):
                    if bin_.shape[0] == 0:
                        continue
                    alpha = 1
                    beta = 0.1
                    local_te = te_bin_priors[i][label_idx]
                    global_te = self.te_preds[:,label_idx].mean()
                    te = local_te*alpha + global_te*(1-alpha)
                    local_tr = tr_bin_priors[i][label_idx]
                    global_tr = self.tr_priors[label_idx]
                    tr = local_tr*beta + global_tr*(1-beta)
                    #local_min = (math.floor(tr * self.n_bins) / self.n_bins)
                    # local_max = local_min + .1
                    # trans = lambda l: min(max((l - local_min) / 1, 0), 1)
                    assert not isomerous_binning, 'not tested'
                    #trans = lambda l: l - local_min
                    # trans = lambda l: l
                    # ratio = (trans(te) / trans(tr))
                    #ratio = np.clip(ratio, 0.1, 2)
                    #ratio = ratio**3
                    #self.te_preds[:, label_idx][bin_] = (te_preds_cp[:, label_idx][bin_]) * ratio
                    old_posterior = te_preds_cp[:, label_idx][bin_]
                    lr = 1
                    #self.te_preds[:, label_idx][bin_] = np.clip(old_posterior + (te-tr)*lr, 0, None)
                    self.te_preds[:, label_idx][bin_] = np.clip(old_posterior + (te - tr) * lr, 0, None)
                    #self.te_preds[:, label_idx][bin_] = (te_preds_cp[:, label_idx][bin_]) * ratio

            # Normalization step
            self.te_preds = (self.te_preds / self.te_preds.sum(axis=1, keepdims=True))
            #self.te_preds = softmax(self.te_preds, axis=1)

            val = np.max(np.abs(te_bin_priors / te_bin_priors_prev) - 1)
            s += 1

        self.iterations = s

        priors = self.te_preds.mean(axis=0)
        posteriors = self.te_preds

        return priors, posteriors

    def __compute_bins_priors(self, bin_priors_placeholder, posteriors, bins, smoothing):
        for label_idx, bins in bins.items():
            for i, bin_ in enumerate(bins):
                if bin_.shape[0] == 0:
                    bin_priors_placeholder[i, label_idx] = smoothing
                    continue
                numerator = posteriors[bin_, label_idx].mean()
                bin_prior = (numerator + smoothing) / (1 + self.n_bins * smoothing)  # normalize priors
                bin_priors_placeholder[i, label_idx] = bin_prior

    def __create_bins(self, training: bool, isomerous_binning: bool):
        bins = {}
        preds = self.tr_preds if training else self.te_preds
        if isomerous_binning:
            for label_idx in range(preds.shape[1]):
                bins[label_idx] = isomerous_bins(label_idx, preds, self.n_bins)
        else:
            intervals = np.linspace(0., 1., num=self.n_bins, endpoint=False)
            for label_idx in range(preds.shape[1]):
                bins_ = isometric_bins(label_idx, preds, intervals)
                bins[label_idx] = [bins_[i] for i in intervals]
        return bins
