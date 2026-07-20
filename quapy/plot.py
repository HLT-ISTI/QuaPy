from collections import defaultdict
import math

from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.pyplot import get_cmap
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.stats import ttest_ind_from_stats

import quapy as qp

plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 18


def binary_diagonal(method_names, true_prevs, estim_prevs, pos_class=1, title=None, show_std=True, legend=True,
                    train_prev=None, savepath=None, method_order=None):
    """
    The diagonal plot displays the predicted prevalence values (along the y-axis) as a function of the true prevalence
    values (along the x-axis). The optimal quantifier is described by the diagonal (0,0)-(1,1) of the plot (hence the
    name). It is convenient for binary quantification problems, though it can be used for multiclass problems by
    indicating which class is to be taken as the positive class. (For multiclass quantification problems, other plots
    like the :meth:`error_by_drift` might be preferable though).

    The format convention is as follows: `method_names`, `true_prevs`, and `estim_prevs` are array-like of the same
    length, with the ith element describing the output of an independent experiment. The elements of `true_prevs`, and
    `estim_prevs` are `ndarrays` with coherent shape for the same experiment. Experiments for the same method on
    different datasets can be used, in which case the method name can appear more than once in `method_names`.

    :param method_names: array-like with the method names for each experiment
    :param true_prevs: array-like with the true prevalence values for each experiment. Each entry is a ndarray of
        shape `(n_samples, n_classes)` components.
    :param estim_prevs: array-like with the estimated prevalence values for each experiment. Each entry is a ndarray of
        shape `(n_samples, n_classes)` components and `n_samples` must coincide with the corresponding entry in
        `true_prevs`.
    :param pos_class: index of the positive class (default 1)
    :param title: the title to be displayed in the plot (default None)
    :param show_std: whether to show standard deviations (represented by color bands). This might be inconvenient
        for cases in which many methods are compared, or when the standard deviations are high -- default True)
    :param legend: whether to display the legend (default True)
    :param train_prev: if indicated (default is None), the training prevalence (for the positive class) is highlighted
        in the plot. This is convenient when all the experiments have been conducted in the same dataset, or in
        datasets with the same training prevalence.
    :param savepath: path where to save the plot. If not indicated (as default), the plot is shown.
    :param method_order: if indicated (default is None), imposes the order in which the methods are processed (i.e.,
        listed in the legend and associated with matplotlib colors).
    :return: returns (fig, ax) matplotlib objects for eventual customisation
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid()
    ax.plot([0, 1], [0, 1], '--k', label='ideal', zorder=1)

    method_names, true_prevs, estim_prevs = _merge(method_names, true_prevs, estim_prevs)

    order = list(zip(method_names, true_prevs, estim_prevs))
    if method_order is not None:
        table = {method_name:[true_prev, estim_prev] for method_name, true_prev, estim_prev in order}
        order  = [(method_name, *table[method_name]) for method_name in method_order]

    NUM_COLORS = len(method_names)
    if NUM_COLORS>10:
        cm = plt.get_cmap('tab20')
        ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    for method, true_prev, estim_prev in order:
        true_prev = true_prev[:,pos_class]
        estim_prev = estim_prev[:,pos_class]

        x_ticks = np.unique(true_prev)
        x_ticks.sort()
        y_ave = np.asarray([estim_prev[true_prev == x].mean() for x in x_ticks])
        y_std = np.asarray([estim_prev[true_prev == x].std() for x in x_ticks])

        ax.errorbar(x_ticks, y_ave, fmt='-', marker='o', label=method, markersize=3, zorder=2)
        if show_std:
            ax.fill_between(x_ticks, y_ave - y_std, y_ave + y_std, alpha=0.25)

    if train_prev is not None:
        train_prev = train_prev[pos_class]
        ax.scatter(train_prev, train_prev, c='c', label='tr-prev', linewidth=2, edgecolor='k', s=100, zorder=3)

    ax.set(xlabel='true prevalence', ylabel='estimated prevalence', title=title)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    if legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    _save_or_show(savepath)
    return fig, ax


def binary_bias_global(method_names, true_prevs, estim_prevs, pos_class=1, title=None, savepath=None):
    """
    Box-plots displaying the global bias (i.e., signed error computed as the estimated value minus the true value)
    for each quantification method with respect to a given positive class.

    The format convention is as follows: `method_names`, `true_prevs`, and `estim_prevs` are array-like of the same
    length, with the ith element describing the output of an independent experiment. The elements of `true_prevs`, and
    `estim_prevs` are `ndarrays` with coherent shape for the same experiment. Experiments for the same method on
    different datasets can be used, in which case the method name can appear more than once in `method_names`.

    :param method_names: array-like with the method names for each experiment
    :param true_prevs: array-like with the true prevalence values for each experiment. Each entry is a ndarray of
        shape `(n_samples, n_classes)` components.
    :param estim_prevs: array-like with the estimated prevalence values for each experiment. Each entry is a ndarray of
        shape `(n_samples, n_classes)` components and `n_samples` must coincide with the corresponding entry in
        `true_prevs`.
    :param pos_class: index of the positive class
    :param title: the title to be displayed in the plot (default None)
    :param savepath: path where to save the plot. If not indicated (as default), the plot is shown.
    :return: returns (fig, ax) matplotlib objects for eventual customisation
    """

    method_names, true_prevs, estim_prevs = _merge(method_names, true_prevs, estim_prevs)

    fig, ax = plt.subplots()
    ax.grid()

    data, labels = [], []
    for method, true_prev, estim_prev in zip(method_names, true_prevs, estim_prevs):
        true_prev = true_prev[:,pos_class]
        estim_prev = estim_prev[:,pos_class]
        data.append(estim_prev-true_prev)
        labels.append(method)

    ax.boxplot(data, labels=labels, patch_artist=False, showmeans=True)
    plt.xticks(rotation=45)
    ax.set(ylabel='error bias', title=title)

    _save_or_show(savepath)

    return fig, ax


def binary_bias_bins(method_names, true_prevs, estim_prevs, pos_class=1, title=None, nbins=5, colormap=cm.tab10,
                     vertical_xticks=False, legend=True, savepath=None):
    """
    Box-plots displaying the local bias (i.e., signed error computed as the estimated value minus the true value)
    for different bins of (true) prevalence of the positive class, for each quantification method.

    The format convention is as follows: `method_names`, `true_prevs`, and `estim_prevs` are array-like of the same
    length, with the ith element describing the output of an independent experiment. The elements of `true_prevs`, and
    `estim_prevs` are `ndarrays` with coherent shape for the same experiment. Experiments for the same method on
    different datasets can be used, in which case the method name can appear more than once in `method_names`.

    :param method_names: array-like with the method names for each experiment
    :param true_prevs: array-like with the true prevalence values for each experiment. Each entry is a ndarray of
        shape `(n_samples, n_classes)` components.
    :param estim_prevs: array-like with the estimated prevalence values for each experiment. Each entry is a ndarray of
        shape `(n_samples, n_classes)` components and `n_samples` must coincide with the corresponding entry in
        `true_prevs`.
    :param pos_class: index of the positive class
    :param title: the title to be displayed in the plot (default None)
    :param nbins: number of bins (default 5)
    :param colormap: the matplotlib colormap to use (default cm.tab10)
    :param vertical_xticks: whether or not to add secondary grid (default is False)
    :param legend: whether or not to display the legend (default is True)
    :param savepath: path where to save the plot. If not indicated (as default), the plot is shown.
    :return: returns (fig, ax) matplotlib objects for eventual customisation
    """
    from pylab import boxplot, plot, setp

    fig, ax = plt.subplots()
    ax.grid()

    method_names, true_prevs, estim_prevs = _merge(method_names, true_prevs, estim_prevs)

    bins = np.linspace(0, 1, nbins+1)
    binwidth = 1/nbins
    data = {}
    for method, true_prev, estim_prev in zip(method_names, true_prevs, estim_prevs):
        true_prev = true_prev[:,pos_class]
        estim_prev = estim_prev[:,pos_class]

        data[method] = []
        inds = np.digitize(true_prev, bins, right=True)
        for ind in range(len(bins)):
            selected = inds==ind
            data[method].append(estim_prev[selected] - true_prev[selected])

    nmethods = len(method_names)
    boxwidth = binwidth/(nmethods+4)
    for i,bin in enumerate(bins[:-1]):
        boxdata = [data[method][i] for method in method_names]
        positions = [bin+(i*boxwidth)+2*boxwidth for i,_ in enumerate(method_names)]
        box = boxplot(boxdata, showmeans=False, positions=positions, widths = boxwidth, sym='+', patch_artist=True)
        for boxid in range(len(method_names)):
            c = colormap.colors[boxid%len(colormap.colors)]
            setp(box['fliers'][boxid], color=c, marker='+', markersize=3., markeredgecolor=c)
            setp(box['boxes'][boxid], color=c)
            setp(box['medians'][boxid], color='k')

    major_xticks_positions, minor_xticks_positions = [], []
    major_xticks_labels, minor_xticks_labels = [], []
    for i,b in enumerate(bins[:-1]):
        major_xticks_positions.append(b)
        minor_xticks_positions.append(b + binwidth / 2)
        major_xticks_labels.append('')
        minor_xticks_labels.append(f'[{bins[i]:.2f}-{bins[i + 1]:.2f})')
    ax.set_xticks(major_xticks_positions)
    ax.set_xticks(minor_xticks_positions, minor=True)
    ax.set_xticklabels(major_xticks_labels)
    ax.set_xticklabels(minor_xticks_labels, minor=True, rotation='vertical' if vertical_xticks else 'horizontal')

    if vertical_xticks:
        # Pad margins so that markers don't get clipped by the axes
        plt.margins(0.2)
        # Tweak spacing to prevent clipping of tick-labels
        plt.subplots_adjust(bottom=0.15)

    if legend:
        # adds the legend to the list hs, initialized with the "ideal" quantifier (one that has 0 bias across all bins. i.e.
        # a line from (0,0) to (1,0). The other elements are simply labelled dot-plots that are to be removed (setting
        # set_visible to False for all but the first element) after the legend has been placed
        hs=[ax.plot([0, 1], [0, 0], '-k', zorder=2)[0]]
        for colorid in range(len(method_names)):
            color=colormap.colors[colorid % len(colormap.colors)]
            h, = plot([0, 0], '-s', markerfacecolor=color, color='k',mec=color, linewidth=1.)
            hs.append(h)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(hs, ['ideal']+method_names, loc='center left', bbox_to_anchor=(1, 0.5))
        [h.set_visible(False) for h in hs[1:]]

    # x-axis and y-axis labels and limits
    ax.set(xlabel='prevalence', ylabel='error bias', title=title)
    ax.set_xlim(0, 1)

    _save_or_show(savepath)

    return fig, ax


def error_by_drift(method_names, true_prevs, estim_prevs, tr_prevs,
                   n_bins=20, error_name='ae', show_std=False,
                   show_density=True,
                   show_legend=True,
                   logscale=False,
                   title=None,
                   vlines=None,
                   method_order=None,
                   savepath=None):
    """
    Plots the error (along the x-axis, as measured in terms of `error_name`) as a function of the train-test shift
    (along the y-axis, as measured in terms of :meth:`quapy.error.ae`). This plot is useful especially for multiclass
    problems, in which "diagonal plots" may be cumbersone, and in order to gain understanding about how methods
    fare in different regions of the prior probability shift spectrum (e.g., in the low-shift regime vs. in the
    high-shift regime).

    The format convention is as follows: `method_names`, `true_prevs`, and `estim_prevs` are array-like of the same
    length, with the ith element describing the output of an independent experiment. The elements of `true_prevs`, and
    `estim_prevs` are `ndarrays` with coherent shape for the same experiment. Experiments for the same method on
    different datasets can be used, in which case the method name can appear more than once in `method_names`.

    :param method_names: array-like with the method names for each experiment
    :param true_prevs: array-like with the true prevalence values for each experiment. Each entry is a ndarray of
        shape `(n_samples, n_classes)` components.
    :param estim_prevs: array-like with the estimated prevalence values for each experiment. Each entry is a ndarray of
        shape `(n_samples, n_classes)` components and `n_samples` must coincide with the corresponding entry in
        `true_prevs`.
    :param tr_prevs: training prevalence of each experiment
    :param n_bins: number of bins in which the y-axis is to be divided (default is 20)
    :param error_name: a string representing the name of an error function (as defined in `quapy.error`, default is "ae")
    :param show_std: whether or not to show standard deviations as color bands (default is False)
    :param show_density: whether or not to display the distribution of experiments for each bin (default is True)
    :param show_density: whether or not to display the legend of the chart (default is True)
    :param logscale: whether or not to log-scale the y-error measure (default is False)
    :param title: title of the plot (default is None)
    :param vlines: array-like list of values (default is None). If indicated, highlights some regions of the space
        using vertical dotted lines.
    :param method_order: if indicated (default is None), imposes the order in which the methods are processed (i.e.,
        listed in the legend and associated with matplotlib colors).
    :param savepath: path where to save the plot. If not indicated (as default), the plot is shown.
    :return: returns (fig, ax) matplotlib objects for eventual customisation
    """

    fig, ax = plt.subplots()
    ax.grid()

    x_error = qp.error.ae
    y_error = getattr(qp.error, error_name)

    if method_order is None:
        method_order = []

    # get all data as a dictionary {'m':{'x':ndarray, 'y':ndarray}} where 'm' is a method name (in the same
    # order as in method_order (if specified), and where 'x' are the train-test shifts (computed as according to
    # x_error function) and 'y' is the estim-test shift (computed as according to y_error)
    data = _join_data_by_drift(method_names, true_prevs, estim_prevs, tr_prevs, x_error, y_error, method_order)

    _set_colors(ax, n_methods=len(method_order))

    bins = np.linspace(0, 1, n_bins+1)
    binwidth = 1 / n_bins
    min_x, max_x, min_y, max_y = None, None, None, None
    npoints = np.zeros(len(bins), dtype=float)
    for method in method_order:
        tr_test_drifts = data[method]['x']
        method_drifts = data[method]['y']
        if logscale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.minorticks_off()

        inds = np.digitize(tr_test_drifts, bins, right=True)

        xs, ys, ystds = [], [], []
        for p,ind in enumerate(range(len(bins))):
            selected = inds==ind
            if selected.sum() > 0:
                xs.append(ind*binwidth-binwidth/2)
                ys.append(np.mean(method_drifts[selected]))
                ystds.append(np.std(method_drifts[selected]))
                npoints[p] += len(method_drifts[selected])

        xs = np.asarray(xs)
        ys = np.asarray(ys)
        ystds = np.asarray(ystds)

        min_x_method, max_x_method, min_y_method, max_y_method = xs.min(), xs.max(), ys.min(), ys.max()
        min_x = min_x_method if min_x is None or min_x_method < min_x else min_x
        max_x = max_x_method if max_x is None or max_x_method > max_x else max_x
        max_y = max_y_method if max_y is None or max_y_method > max_y else max_y
        min_y = min_y_method if min_y is None or min_y_method < min_y else min_y
        max_y = max_y_method if max_y is None or max_y_method > max_y else max_y

        ax.errorbar(xs, ys, fmt='-', marker='o', color='w', markersize=8, linewidth=4, zorder=1)
        ax.errorbar(xs, ys, fmt='-', marker='o', label=method, markersize=6, linewidth=2, zorder=2)

        if show_std:
            ax.fill_between(xs, ys-ystds, ys+ystds, alpha=0.25)

    if show_density:
        ax2 = ax.twinx()
        densities = npoints/np.sum(npoints)
        ax2.bar([ind * binwidth-binwidth/2 for ind in range(len(bins))],
               densities, alpha=0.15, color='g', width=binwidth, label='density')
        ax2.set_ylim(0,max(densities))
        ax2.spines['right'].set_color('g')
        ax2.tick_params(axis='y', colors='g')
    
    ax.set(xlabel=f'Prior shift between training set and test sample',
           ylabel=f'{error_name.upper()} (true prev, predicted prev)',
           title=title)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    if vlines:
        for vline in vlines:
            ax.axvline(vline, 0, 1, linestyle='--', color='k')

    ax.set_xlim(min_x, max_x)
    if logscale:
        #nice scale for the logaritmic axis
        ax.set_ylim(0,10 ** math.ceil(math.log10(max_y)))
    
    if show_legend:
        fig.legend(loc='center left',
                  bbox_to_anchor=(1, 0.5),
                  ncol=1)

    _save_or_show(savepath)

    return fig, ax


def brokenbar_supremacy_by_drift(method_names, true_prevs, estim_prevs, tr_prevs,
                                 n_bins=20, binning='isomerous',
                                 x_error='ae', y_error='ae', ttest_alpha=0.005, tail_density_threshold=0.005,
                                 method_order=None,
                                 savepath=None):
    """
    Displays (only) the top performing methods for different regions of the train-test shift in form of a broken
    bar chart, in which each method has bars only for those regions in which either one of the following conditions
    hold: (i) it is the best method (in average) for the bin, or (ii) it is not statistically significantly different
    (in average) as according to a two-sided t-test on independent samples at confidence `ttest_alpha`.
    The binning can be made "isometric" (same size), or "isomerous" (same number of experiments -- default). A second
    plot is displayed on top, that displays the distribution of experiments for each bin (when binning="isometric") or
    the percentiles points of the distribution (when binning="isomerous").

    The format convention is as follows: `method_names`, `true_prevs`, and `estim_prevs` are array-like of the same
    length, with the ith element describing the output of an independent experiment. The elements of `true_prevs`, and
    `estim_prevs` are `ndarrays` with coherent shape for the same experiment. Experiments for the same method on
    different datasets can be used, in which case the method name can appear more than once in `method_names`.

    :param method_names: array-like with the method names for each experiment
    :param true_prevs: array-like with the true prevalence values for each experiment. Each entry is a ndarray of
        shape `(n_samples, n_classes)` components.
    :param estim_prevs: array-like with the estimated prevalence values for each experiment. Each entry is a ndarray of
        shape `(n_samples, n_classes)` components and `n_samples` must coincide with the corresponding entry in
        `true_prevs`.
    :param tr_prevs: training prevalence of each experiment
    :param n_bins: number of bins in which the y-axis is to be divided (default is 20)
    :param binning: type of binning, either "isomerous" (default) or "isometric"
    :param x_error: a string representing the name of an error function (as defined in `quapy.error`) to be used for
        measuring the amount of train-test shift (default is "ae")
    :param y_error: a string representing the name of an error function (as defined in `quapy.error`) to be used for
        measuring the amount of error in the prevalence estimations (default is "ae")
    :param ttest_alpha: the confidence interval above which a p-value (two-sided t-test on independent samples) is
        to be considered as an indicator that the two means are not statistically significantly different. Default is
        0.005, meaning that a `p-value > 0.005` indicates the two methods involved are to be considered similar
    :param tail_density_threshold: sets a threshold on the density of experiments (over the total number of experiments)
        below which a bin in the tail (i.e., the right-most ones) will be discarded. This is in order to avoid some
        bins to be shown for train-test outliers.
    :param method_order: if indicated (default is None), imposes the order in which the methods are processed (i.e.,
        listed in the legend and associated with matplotlib colors).
    :param savepath: path where to save the plot. If not indicated (as default), the plot is shown.
    :return: returns (fig, ax) matplotlib objects for eventual customisation
    """
    assert binning in ['isomerous', 'isometric'], 'unknown binning type; valid types are "isomerous" and "isometric"'

    x_error = getattr(qp.error, x_error)
    y_error = getattr(qp.error, y_error)

    if method_order is None:
        method_order = []

    # get all data as a dictionary {'m':{'x':ndarray, 'y':ndarray}} where 'm' is a method name (in the same
    # order as in method_order (if specified), and where 'x' are the train-test shifts (computed as according to
    # x_error function) and 'y' is the estim-test shift (computed as according to y_error)
    data = _join_data_by_drift(method_names, true_prevs, estim_prevs, tr_prevs, x_error, y_error, method_order)

    if method_order is None:
        method_order = method_names

    if binning == 'isomerous':
        # take bins containing the same amount of examples
        tr_test_drifts = np.concatenate([data[m]['x'] for m in method_order])
        bins = np.quantile(tr_test_drifts, q=np.linspace(0, 1, n_bins+1)).flatten()
    else:
        # take equidistant bins
        bins = np.linspace(0, 1, n_bins+1)
    bins[0] = -0.001
    bins[-1] += 0.001

    # we use this to keep track of how many datapoits contribute to each bin
    inds_histogram_global = np.zeros(n_bins, dtype=float)
    n_methods = len(method_order)
    buckets = np.zeros(shape=(n_methods, n_bins, 3))
    for i, method in enumerate(method_order):
        tr_test_drifts = data[method]['x']
        method_drifts = data[method]['y']

        inds = np.digitize(tr_test_drifts, bins, right=False)
        inds_histogram_global += np.histogram(tr_test_drifts, density=False, bins=bins)[0]

        for j in range(len(bins)):
            selected = inds == j
            if selected.sum() > 0:
                buckets[i, j-1, 0] = np.mean(method_drifts[selected])
                buckets[i, j-1, 1] = np.std(method_drifts[selected])
                buckets[i, j-1, 2] = selected.sum()

    # cancel last buckets with low density
    histogram = inds_histogram_global / inds_histogram_global.sum()
    for tail in reversed(range(len(histogram))):
        if histogram[tail] < tail_density_threshold:
            buckets[:,tail,2] = 0
        else:
            break

    salient_methods = set()
    best_methods = []
    for bucket in range(buckets.shape[1]):
        nc = buckets[:, bucket, 2].sum()
        if nc == 0:
            best_methods.append([])
            continue

        order = np.argsort(buckets[:, bucket, 0])
        rank1 = order[0]
        best_bucket_methods = [method_order[rank1]]
        best_mean, best_std, best_nc = buckets[rank1, bucket, :]
        for method_index in order[1:]:
            method_mean, method_std, method_nc = buckets[method_index, bucket, :]
            _, pval = ttest_ind_from_stats(best_mean, best_std, best_nc, method_mean, method_std, method_nc)
            if pval > ttest_alpha:
                best_bucket_methods.append(method_order[method_index])
        best_methods.append(best_bucket_methods)
        salient_methods.update(best_bucket_methods)

    if binning=='isomerous':
        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [0.2, 1]}, figsize=(20, len(salient_methods)))
    else:
        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]}, figsize=(20, len(salient_methods)))

    ax = axes[1]
    high_from = 0
    yticks, yticks_method_names = [], []
    color = get_cmap('Accent').colors
    vlines = []
    bar_high = 1
    for method in [m for m in method_order if m in salient_methods]:
        broken_paths = []
        path_start, path_end = None, None
        for i, best_bucket_methods in enumerate(best_methods):
            if method in best_bucket_methods:
                if path_start is None:
                    path_start = bins[i]
                    path_end = bins[i+1]-path_start
                else:
                    path_end += bins[i+1]-bins[i]
            else:
                if path_start is not None:
                    broken_paths.append(tuple((path_start, path_end)))
                    path_start, path_end = None, None
        if path_start is not None:
            broken_paths.append(tuple((path_start, path_end)))

        ax.broken_barh(broken_paths, (high_from, bar_high), facecolors=color[len(yticks_method_names)])
        yticks.append(high_from+bar_high/2)
        high_from += bar_high
        yticks_method_names.append(method)
        for path_start, path_end in broken_paths:
            vlines.extend([path_start, path_start+path_end])

    vlines = np.unique(vlines)
    vlines = sorted(vlines)
    for v in vlines[1:-1]:
        ax.axvline(x=v, color='k', linestyle='--')

    ax.set_ylim(0, high_from)
    ax.set_xlim(vlines[0], vlines[-1])
    ax.set_xlabel('Distribution shift between training set and sample')

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks_method_names)

    # upper plot (explaining distribution)
    ax = axes[0]
    if binning == 'isometric':
        # show the density for each region
        bins[0]=0
        y_pos = [b+(bins[i+1]-b)/2 for i,b in enumerate(bins[:-1]) if histogram[i]>0]
        bar_width = [bins[i+1]-bins[i] for i in range(len(bins[:-1])) if histogram[i]>0]
        ax.bar(y_pos, [n for n in histogram if n>0], bar_width, align='center', alpha=0.5, color='silver')
        ax.set_ylabel('shift\ndistribution', rotation=0, ha='right', va='center')
        ax.set_xlim(vlines[0], vlines[-1])
        ax.get_xaxis().set_visible(False)
        plt.subplots_adjust(wspace=0, hspace=0.1)
    else:
        # show the percentiles of the distribution
        cumsum = np.cumsum(histogram)
        for i in range(len(bins[:-1])):
            start, width = bins[i], bins[i+1]-bins[i]
            ax.broken_barh([tuple((start, width))], (0, 1), facecolors='whitesmoke' if i%2==0 else 'silver')
            if i < len(bins)-2:
                ax.text(bins[i+1], 0.5, '$P_{'+f'{int(np.round(cumsum[i]*100))}'+'}$', ha='center')
        ax.set_ylim(0, 1)
        ax.set_xlim(vlines[0], vlines[-1])
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        plt.subplots_adjust(wspace=0, hspace=0)

    _save_or_show(savepath)

    return fig, ax


def plot_simplex(
        point_layers=None,
        region_layers=None,
        density_function=None,
        density_color='#1f77b4',
        density_alpha=1.0,
        resolution=400,
        class_names=None,
        title=None,
        show_legend=True,
        legend_loc='lower center',
        legend_bbox_to_anchor=(0.5, -0.08),
        legend_ncol=2,
        figsize=(6.8, 6.2),
        class_name_fontsize=10,
        title_fontsize=11,
        legend_fontsize=9,
        ax=None,
        savepath=None):
    """
    Plots data on the ternary simplex for three-class quantification problems.

    This utility is convenient for visualising prevalence vectors, posterior triplets,
    confidence regions, or any other points that lie on the 2-dimensional probability
    simplex. The plot can combine three optional layer types:

    * `point_layers`: scatter layers for one or more prevalence clouds or reference points
    * `region_layers`: shaded regions defined by callables on prevalence vectors
    * `density_function`: a scalar function evaluated on the simplex and rendered as a heatmap

    Each entry in `point_layers` is a dictionary with a mandatory `points` field
    containing an array-like of shape `(n_points, 3)` or `(3,)`. Optional fields are
    `label` for the legend and `style` for matplotlib scatter keyword arguments.

    Each entry in `region_layers` is a dictionary with a mandatory `fn` field containing
    a callable that receives prevalence vectors and returns region-membership scores.
    Optional fields are `label`, `color`, and `alpha`.

    :param point_layers: optional list of point-layer dictionaries
    :param region_layers: optional list of region-layer dictionaries
    :param density_function: optional callable receiving prevalence vectors and returning
        scalar values
    :param density_color: color used for the density heatmap
    :param density_alpha: opacity for the density heatmap
    :param resolution: number of grid steps per axis used for rendering regions and densities
    :param class_names: optional list or tuple with the three class names
    :param title: optional plot title
    :param show_legend: whether to display the legend
    :param legend_loc: location string passed to matplotlib for the legend
    :param legend_bbox_to_anchor: optional legend anchor box
    :param legend_ncol: number of legend columns
    :param figsize: figure size used when `ax` is not provided
    :param class_name_fontsize: fontsize used for simplex vertex labels
    :param title_fontsize: fontsize used for the optional title
    :param legend_fontsize: fontsize used for the legend
    :param ax: optional matplotlib axes object; if not provided, a new figure is created
    :param savepath: path where to save the plot; if not indicated, the plot is shown when
        `ax` is not provided
    :return: returns `(fig, ax)` matplotlib objects for eventual customisation
    """
    if class_names is None:
        class_names = ('Y=1', 'Y=2', 'Y=3')
    if len(class_names) != 3:
        raise ValueError(f'expected exactly 3 class names, got {len(class_names)}')

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if density_function is not None:
        _plot_simplex_density(ax, density_function, resolution, density_color, density_alpha)

    if region_layers:
        _plot_simplex_regions(ax, region_layers, resolution)

    if point_layers:
        _plot_simplex_points(ax, point_layers)

    simplex_ymax = np.sqrt(3) / 2
    triangle = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, simplex_ymax],
        [0.0, 0.0],
    ])
    ax.plot(triangle[:, 0], triangle[:, 1], color='black')

    ax.text(-0.05, -0.05, class_names[0], ha='right', va='top', fontsize=class_name_fontsize)
    ax.text(1.05, -0.05, class_names[1], ha='left', va='top', fontsize=class_name_fontsize)
    ax.text(0.5, simplex_ymax + 0.05, class_names[2], ha='center', va='bottom', fontsize=class_name_fontsize)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    ax.set_aspect('equal')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, simplex_ymax + 0.1)
    ax.axis('off')

    if show_legend:
        _, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, ncol=legend_ncol, fontsize=legend_fontsize, frameon=False)

    fig.tight_layout(pad=0.6)

    if savepath is not None:
        qp.util.create_parent_dir(savepath)
        fig.savefig(savepath, bbox_inches='tight')
    elif own_figure:
        plt.show()

    return fig, ax


def _merge(method_names, true_prevs, estim_prevs):
    ndims = true_prevs[0].shape[1]
    data = defaultdict(lambda: {'true': np.empty(shape=(0, ndims)), 'estim': np.empty(shape=(0, ndims))})
    method_order=[]
    for method, true_prev, estim_prev in zip(method_names, true_prevs, estim_prevs):
        data[method]['true'] = np.concatenate([data[method]['true'], true_prev])
        data[method]['estim'] = np.concatenate([data[method]['estim'], estim_prev])
        if method not in method_order:
            method_order.append(method)
    true_prevs_ = [data[m]['true'] for m in method_order]
    estim_prevs_ = [data[m]['estim'] for m in method_order]
    return method_order, true_prevs_, estim_prevs_


def _set_colors(ax, n_methods):
    NUM_COLORS = n_methods
    if NUM_COLORS>10:
        cm = plt.get_cmap('tab20')
        ax.set_prop_cycle(color=[cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])


def _save_or_show(savepath):
    # if savepath is specified, then saves the plot in that path; otherwise the plot is shown
    if savepath is not None:
        qp.util.create_parent_dir(savepath)
        # plt.tight_layout()
        plt.savefig(savepath, bbox_inches='tight')
    else:
        plt.show()


def _join_data_by_drift(method_names, true_prevs, estim_prevs, tr_prevs, x_error, y_error, method_order):
    data = defaultdict(lambda: {'x': np.empty(shape=(0)), 'y': np.empty(shape=(0))})

    if method_order is None:
        method_order = []

    for method, test_prevs_i, estim_prevs_i, tr_prev_i in zip(method_names, true_prevs, estim_prevs, tr_prevs):
        tr_prev_i = np.repeat(tr_prev_i.reshape(1, -1), repeats=test_prevs_i.shape[0], axis=0)

        tr_test_drifts = x_error(test_prevs_i, tr_prev_i)
        data[method]['x'] = np.concatenate([data[method]['x'], tr_test_drifts])

        method_drifts = y_error(test_prevs_i, estim_prevs_i)
        data[method]['y'] = np.concatenate([data[method]['y'], method_drifts])

        if method not in method_order:
            method_order.append(method)

    return data


def _simplex_to_cartesian(prevalences):
    prevalences = np.asarray(prevalences, dtype=float)
    prevalences = np.atleast_2d(prevalences)
    if prevalences.shape[1] != 3:
        raise ValueError(f'plot_simplex expects prevalence vectors of shape (_, 3); found {prevalences.shape}')
    x = prevalences[:, 1] + 0.5 * prevalences[:, 2]
    y = prevalences[:, 2] * (np.sqrt(3) / 2)
    return x, y


def _barycentric_from_xy(x, y):
    p3 = 2 * y / np.sqrt(3)
    p2 = x - 0.5 * p3
    p1 = 1 - p2 - p3
    return np.stack([p1, p2, p3], axis=-1)


def _simplex_mesh(resolution):
    simplex_ymax = np.sqrt(3) / 2
    xs = np.linspace(0, 1, resolution)
    ys = np.linspace(0, simplex_ymax, resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts_bary = _barycentric_from_xy(grid_x, grid_y)
    mask = np.all(pts_bary >= 0, axis=-1)
    return xs, ys, pts_bary, mask


def _evaluate_simplex_function(function, points):
    points = np.asarray(points, dtype=float)
    try:
        values = np.asarray(function(points), dtype=float)
        if values.shape == (points.shape[0],):
            return values
        if values.shape == points.shape[:-1]:
            return values.reshape(-1)
    except Exception:
        pass
    return np.asarray([function(point) for point in points], dtype=float)


def _region_colormap(color='blue', alpha=0.35):
    return ListedColormap([
        (1.0, 1.0, 1.0, 0.0),
        (*mcolors.to_rgb(color), alpha),
    ])


def _plot_simplex_points(ax, point_layers):
    for layer in point_layers:
        points = np.asarray(layer['points'], dtype=float)
        style = {'s': 25, 'alpha': 0.8}
        style.update(layer.get('style', {}))
        ax.scatter(*_simplex_to_cartesian(points), label=layer.get('label'), **style)


def _plot_simplex_regions(ax, region_layers, resolution):
    xs, ys, pts_bary, simplex_mask = _simplex_mesh(resolution)
    valid_points = pts_bary[simplex_mask]

    for layer in region_layers:
        mask = np.zeros(simplex_mask.shape, dtype=float)
        values = _evaluate_simplex_function(layer['fn'], valid_points)
        mask[simplex_mask] = values
        ax.pcolormesh(
            xs,
            ys,
            mask,
            shading='auto',
            cmap=_region_colormap(layer.get('color', 'blue'), layer.get('alpha', 0.35)),
        )
        if layer.get('label') is not None:
            ax.scatter([], [], color=layer.get('color', 'blue'), alpha=layer.get('alpha', 0.35), label=layer['label'])


def _plot_simplex_density(ax, density_function, resolution, color, alpha):
    xs, ys, pts_bary, simplex_mask = _simplex_mesh(resolution)
    valid_points = pts_bary[simplex_mask]
    density = np.full(simplex_mask.shape, np.nan, dtype=float)
    values = _evaluate_simplex_function(density_function, valid_points)
    min_v, max_v = np.min(values), np.max(values)
    if max_v > min_v:
        values = (values - min_v) / (max_v - min_v)
    density[simplex_mask] = values

    cmap = LinearSegmentedColormap.from_list('simplex_density', ['white', color])
    ax.pcolormesh(xs, ys, density, shading='auto', cmap=cmap, alpha=alpha)


def calibration_plot(prob_classifier, X, y, nbins=10, savepath=None):
    posteriors = prob_classifier.predict_proba(X)
    assert posteriors.ndim==2, 'calibration plot only works for binary problems'
    posteriors = posteriors[:,1]
    pred_y = posteriors>=0.5
    bins = np.linspace(0, 1, nbins + 1)
    binned_values = np.digitize(posteriors, bins, right=False)
    correct = pred_y == y
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bins_names = np.arange(nbins)
    y_axis = [correct[binned_values==bin].mean() for bin in bins_names]
    y_axis = [v if not np.isnan(v) else 0 for v in y_axis]
    # Crear el gráfico de barras
    plt.bar(bin_centers, y_axis, width=bins[1]-bins[0], edgecolor='black', alpha=0.7)

    # Etiquetas y título
    plt.xlabel("Bin")
    plt.ylabel("Value")
    plt.title("Bar plot of calculated values per bin")
    plt.xticks(bin_centers, [f"{b:.2f}" for b in bin_centers], rotation=45)

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import quapy as qp
    from sklearn.linear_model import LogisticRegression
    data = qp.datasets.fetch_UCIBinaryDataset(qp.datasets.UCI_BINARY_DATASETS[6])
    train, test = data.train_test
    classifier = LogisticRegression()
    classifier.fit(*train.Xy)
    calibration_plot(classifier, *test.Xy)
