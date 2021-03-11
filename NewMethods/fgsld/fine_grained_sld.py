import numpy as np
from metrics import isomerous_bins, isometric_bins
from em import History, get_measures_single_history
from sklearn.model_selection import cross_val_predict
import math


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

    def run(self, isomerous_binning, epsilon=1e-6, compute_bins_at_every_iter=True, return_posteriors_hist=False):
        """
        Run the FGSLD algorithm.

        :param isomerous_binning: whether to use isomerous or isometric binning.
        :param epsilon: stopping condition.
        :param compute_bins_at_every_iter: whether FGSLD should recompute the posterior bins at every iteration or not.
        :param return_posteriors_hist: whether to return posteriors at every iteration or not.
        :return: If `return_posteriors_hist` is true, the returned posteriors will be a list of numpy arrays, else a single numpy array with posteriors at last iteration.
        """
        smoothing_tr = 1 / (2 * self.tr_preds.shape[0])
        smoothing_te = 1 / (2 * self.te_preds.shape[0])
        s = 0
        tr_bin_priors = np.zeros((self.n_bins, self.tr_preds.shape[1]), dtype=np.float)
        te_bin_priors = np.zeros((self.n_bins, self.te_preds.shape[1]), dtype=np.float)
        tr_bins = self.__create_bins(training=True, isomerous_binning=isomerous_binning)
        te_bins = self.__create_bins(training=False, isomerous_binning=isomerous_binning)
        self.__compute_bins_priors(tr_bin_priors, self.tr_preds, tr_bins, smoothing_tr)

        val = 2 * epsilon
        if return_posteriors_hist:
            posteriors_hist = [self.te_preds.copy()]
        while not val < epsilon and s < 1000:
            assert np.all(np.around(self.te_preds.sum(axis=1), 4) == 1), f"Probabilities do not sum to 1:\ns={s}, " \
                                                                         f"probs={self.te_preds.sum(axis=1)}"
            if compute_bins_at_every_iter:
                te_bins = self.__create_bins(training=False, isomerous_binning=isomerous_binning)

            if s == 0:
                te_bin_priors_prev = tr_bin_priors.copy()
            else:
                te_bin_priors_prev = te_bin_priors.copy()
            self.__compute_bins_priors(te_bin_priors, self.te_preds, te_bins, smoothing_te)

            te_preds_cp = self.te_preds.copy()
            for label_idx, bins in te_bins.items():
                for i, bin_ in enumerate(bins):
                    if bin_.shape[0] == 0:
                        continue
                    te = te_bin_priors[i][label_idx]
                    tr = tr_bin_priors[i][label_idx]
                    # local_min = (math.floor(tr * 10) / 10)
                    # local_max = local_min + .1
                    # trans = lambda l: min(max((l - local_min) / 1, 0), 1)
                    trans = lambda l: l
                    self.te_preds[:, label_idx][bin_] = (te_preds_cp[:, label_idx][bin_]) * \
                                                        (trans(te) / trans(tr))

            # Normalization step
            self.te_preds = (self.te_preds / self.te_preds.sum(axis=1, keepdims=True))

            val = 0
            for label_idx in range(te_bin_priors.shape[1]):
                temp = max(abs((te_bin_priors[:, label_idx] / te_bin_priors_prev[:, label_idx]) - 1))
                if temp > val:
                    val = temp
            s += 1
            if return_posteriors_hist:
                posteriors_hist.append(self.te_preds.copy())
        if return_posteriors_hist:
            return self.te_preds.mean(axis=0), posteriors_hist
        return self.te_preds.mean(axis=0), self.te_preds

    def __compute_bins_priors(self, bin_priors_placeholder, posteriors, bins, smoothing):
        for label_idx, bins in bins.items():
            for i, bin_ in enumerate(bins):
                if bin_.shape[0] == 0:
                    bin_priors_placeholder[i, label_idx] = smoothing
                    continue
                numerator = posteriors[:, label_idx][bin_].mean()
                bin_prior = (numerator + smoothing) / (1 + self.n_bins * smoothing)  # normalize priors
                bin_priors_placeholder[i, label_idx] = bin_prior

    def __find_bin_idx(self, label_bins: [np.array], idx: int or list):
        if hasattr(idx, '__len__'):
            idxs = np.zeros(len(idx), dtype=np.int)
            for i, bin_ in enumerate(label_bins):
                for j, id_ in enumerate(idx):
                    if id_ in bin_:
                        idxs[j] = i
            return idxs
        else:
            for i, bin_ in enumerate(label_bins):
                if idx in bin_:
                    return i

    def __create_bins(self, training: bool, isomerous_binning: bool):
        bins = {}
        preds = self.tr_preds if training else self.te_preds
        if isomerous_binning:
            for label_idx in range(preds.shape[1]):
                bins[label_idx] = isomerous_bins(label_idx, preds, self.n_bins)
        else:
            intervals = np.linspace(0., 1., num=self.n_bins, endpoint=False)
            for label_idx in range(preds.shape[1]):
                bins_ = isometric_bins(label_idx, preds, intervals, 0.1)
                bins[label_idx] = [bins_[i] for i in intervals]
        return bins
