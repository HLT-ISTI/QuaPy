from scipy.sparse import csc_matrix, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
import numpy as np
from joblib import Parallel, delayed
import sklearn
import math
from scipy.stats import t


class ContTable:
    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        self.tp=tp
        self.tn=tn
        self.fp=fp
        self.fn=fn

    def get_d(self): return self.tp + self.tn + self.fp + self.fn

    def get_c(self): return self.tp + self.fn

    def get_not_c(self): return self.tn + self.fp

    def get_f(self): return self.tp + self.fp

    def get_not_f(self): return self.tn + self.fn

    def p_c(self): return (1.0*self.get_c())/self.get_d()

    def p_not_c(self): return 1.0-self.p_c()

    def p_f(self): return (1.0*self.get_f())/self.get_d()

    def p_not_f(self): return 1.0-self.p_f()

    def p_tp(self): return (1.0*self.tp) / self.get_d()

    def p_tn(self): return (1.0*self.tn) / self.get_d()

    def p_fp(self): return (1.0*self.fp) / self.get_d()

    def p_fn(self): return (1.0*self.fn) / self.get_d()

    def tpr(self):
        c = 1.0*self.get_c()
        return self.tp / c if c > 0.0 else 0.0

    def fpr(self):
        _c = 1.0*self.get_not_c()
        return self.fp / _c if _c > 0.0 else 0.0


def __ig_factor(p_tc, p_t, p_c):
    den = p_t * p_c
    if den != 0.0 and p_tc != 0:
        return p_tc * math.log(p_tc / den, 2)
    else:
        return 0.0


def information_gain(cell):
    return __ig_factor(cell.p_tp(), cell.p_f(), cell.p_c()) + \
           __ig_factor(cell.p_fp(), cell.p_f(), cell.p_not_c()) +\
           __ig_factor(cell.p_fn(), cell.p_not_f(), cell.p_c()) + \
           __ig_factor(cell.p_tn(), cell.p_not_f(), cell.p_not_c())


def squared_information_gain(cell):
    return information_gain(cell)**2


def posneg_information_gain(cell):
    ig = information_gain(cell)
    if cell.tpr() < cell.fpr():
        return -ig
    else:
        return ig


def pos_information_gain(cell):
    if cell.tpr() < cell.fpr():
        return 0
    else:
        return information_gain(cell)

def pointwise_mutual_information(cell):
    return __ig_factor(cell.p_tp(), cell.p_f(), cell.p_c())


def gss(cell):
    return cell.p_tp()*cell.p_tn() - cell.p_fp()*cell.p_fn()


def chi_square(cell):
    den = cell.p_f() * cell.p_not_f() * cell.p_c() * cell.p_not_c()
    if den==0.0: return 0.0
    num = gss(cell)**2
    return num / den


def conf_interval(xt, n):
    if n>30:
        z2 = 3.84145882069 # norm.ppf(0.5+0.95/2.0)**2
    else:
        z2 = t.ppf(0.5 + 0.95 / 2.0, df=max(n-1,1)) ** 2
    p = (xt + 0.5 * z2) / (n + z2)
    amplitude = 0.5 * z2 * math.sqrt((p * (1.0 - p)) / (n + z2))
    return p, amplitude


def strength(minPosRelFreq, minPos, maxNeg):
    if minPos > maxNeg:
        return math.log(2.0 * minPosRelFreq, 2.0)
    else:
        return 0.0


#set cancel_features=True to allow some features to be weighted as 0 (as in the original article)
#however, for some extremely imbalanced dataset caused all documents to be 0
def conf_weight(cell, cancel_features=False):
    c = cell.get_c()
    not_c = cell.get_not_c()
    tp = cell.tp
    fp = cell.fp

    pos_p, pos_amp = conf_interval(tp, c)
    neg_p, neg_amp = conf_interval(fp, not_c)

    min_pos = pos_p-pos_amp
    max_neg = neg_p+neg_amp
    den = (min_pos + max_neg)
    minpos_relfreq = min_pos / (den if den != 0 else 1)

    str_tplus = strength(minpos_relfreq, min_pos, max_neg);

    if str_tplus == 0 and not cancel_features:
        return 1e-20

    return str_tplus


def get_tsr_matrix(cell_matrix, tsr_score_funtion):
    nC = len(cell_matrix)
    nF = len(cell_matrix[0])
    tsr_matrix = [[tsr_score_funtion(cell_matrix[c,f]) for f in range(nF)] for c in range(nC)]
    return np.array(tsr_matrix)


def feature_label_contingency_table(positive_document_indexes, feature_document_indexes, nD):
    tp_ = len(positive_document_indexes & feature_document_indexes)
    fp_ = len(feature_document_indexes - positive_document_indexes)
    fn_ = len(positive_document_indexes - feature_document_indexes)
    tn_ = nD - (tp_ + fp_ + fn_)
    return ContTable(tp=tp_, tn=tn_, fp=fp_, fn=fn_)


def category_tables(feature_sets, category_sets, c, nD, nF):
    return [feature_label_contingency_table(category_sets[c], feature_sets[f], nD) for f in range(nF)]


def get_supervised_matrix(coocurrence_matrix, label_matrix, n_jobs=-1):
    """
    Computes the nC x nF supervised matrix M where Mcf is the 4-cell contingency table for feature f and class c.
    Efficiency O(nF x nC x log(S)) where S is the sparse factor
    """

    nD, nF = coocurrence_matrix.shape
    nD2, nC = label_matrix.shape

    if nD != nD2:
        raise ValueError('Number of rows in coocurrence matrix shape %s and label matrix shape %s is not consistent' %
                         (coocurrence_matrix.shape,label_matrix.shape))

    def nonzero_set(matrix, col):
        return set(matrix[:, col].nonzero()[0])

    if isinstance(coocurrence_matrix, csr_matrix):
        coocurrence_matrix = csc_matrix(coocurrence_matrix)
    feature_sets = [nonzero_set(coocurrence_matrix, f) for f in range(nF)]
    category_sets = [nonzero_set(label_matrix, c) for c in range(nC)]
    cell_matrix = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(category_tables)(feature_sets, category_sets, c, nD, nF) for c in range(nC)
    )
    return np.array(cell_matrix)


class TSRweighting(BaseEstimator,TransformerMixin):
    """
    Supervised Term Weighting function based on any Term Selection Reduction (TSR) function (e.g., information gain,
    chi-square, etc.) or, more generally, on any function that could be computed on the 4-cell contingency table for
    each category-feature pair.
    The supervised_4cell_matrix is a `(n_classes, n_words)` matrix containing the 4-cell contingency tables
    for each class-word pair, and can be pre-computed (e.g., during the feature selection phase) and passed as an
    argument.
    When `n_classes>1`, i.e., in multiclass scenarios, a global_policy is used in order to determine a
    single feature-score which informs about its relevance. Accepted policies include "max" (takes the max score
    across categories), "ave" and "wave" (take the average, or weighted average, across all categories -- weights
    correspond to the class prevalence), and "sum" (which sums all category scores).
    """

    def __init__(self, tsr_function, global_policy='max', supervised_4cell_matrix=None, sublinear_tf=True, norm='l2', min_df=3, n_jobs=-1):
        if global_policy not in ['max', 'ave', 'wave', 'sum']: raise ValueError('Global policy should be in {"max", "ave", "wave", "sum"}')
        self.tsr_function = tsr_function
        self.global_policy = global_policy
        self.supervised_4cell_matrix = supervised_4cell_matrix
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.min_df = min_df
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.count_vectorizer = CountVectorizer(min_df=self.min_df)
        X = self.count_vectorizer.fit_transform(X)

        self.tf_vectorizer = TfidfTransformer(
            norm=None, use_idf=False, smooth_idf=False, sublinear_tf=self.sublinear_tf
        ).fit(X)

        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)

        nD, nC = y.shape
        nF = len(self.tf_vectorizer.get_feature_names_out())

        if self.supervised_4cell_matrix is None:
            self.supervised_4cell_matrix = get_supervised_matrix(X, y, n_jobs=self.n_jobs)
        else:
            if self.supervised_4cell_matrix.shape != (nC, nF):
                raise ValueError("Shape of supervised information matrix is inconsistent with X and y")

        tsr_matrix = get_tsr_matrix(self.supervised_4cell_matrix, self.tsr_function)

        if self.global_policy == 'ave':
            self.global_tsr_vector = np.average(tsr_matrix, axis=0)
        elif self.global_policy == 'wave':
            category_prevalences = [sum(y[:,c])*1.0/nD for c in range(nC)]
            self.global_tsr_vector = np.average(tsr_matrix, axis=0, weights=category_prevalences)
        elif self.global_policy == 'sum':
            self.global_tsr_vector = np.sum(tsr_matrix, axis=0)
        elif self.global_policy == 'max':
            self.global_tsr_vector = np.amax(tsr_matrix, axis=0)
        return self

    def fit_transform(self, X, y):
        return self.fit(X,y).transform(X)

    def transform(self, X):
        if not hasattr(self, 'global_tsr_vector'): raise NameError('TSRweighting: transform method called before fit.')
        X = self.count_vectorizer.transform(X)
        tf_X = self.tf_vectorizer.transform(X).toarray()
        weighted_X = np.multiply(tf_X, self.global_tsr_vector)
        if self.norm is not None and self.norm!='none':
            weighted_X = sklearn.preprocessing.normalize(weighted_X, norm=self.norm, axis=1, copy=False)
        return csr_matrix(weighted_X)
