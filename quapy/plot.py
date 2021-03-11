from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import quapy as qp
from matplotlib.font_manager import FontProperties

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 16


def binary_diagonal(method_names, true_prevs, estim_prevs, pos_class=1, title=None, show_std=True, legend=True,
                    train_prev=None, savepath=None):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid()
    ax.plot([0, 1], [0, 1], '--k', label='ideal', zorder=1)

    method_names, true_prevs, estim_prevs = _merge(method_names, true_prevs, estim_prevs)

    for method, true_prev, estim_prev in zip(method_names, true_prevs, estim_prevs):
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
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    save_or_show(savepath)


def binary_bias_global(method_names, true_prevs, estim_prevs, pos_class=1, title=None, savepath=None):
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

    save_or_show(savepath)


def binary_bias_bins(method_names, true_prevs, estim_prevs, pos_class=1, title=None, nbins=5, colormap=cm.tab10,
                     vertical_xticks=False, legend=True, savepath=None):
    from pylab import boxplot, plot, setp

    fig, ax = plt.subplots()
    ax.grid()

    method_names, true_prevs, estim_prevs = _merge(method_names, true_prevs, estim_prevs)

    bins = np.linspace(0, 1, nbins+1)
    binwidth = 1/nbins
    data = {}
    for method, true_prev, estim_prev in zip(method_names, true_prevs, estim_prevs):
        true_prev = true_prev[:, pos_class]
        estim_prev = estim_prev[:, pos_class]

        data[method] = []
        inds = np.digitize(true_prev, bins[1:], right=True)
        for ind in range(len(bins)):
            selected = inds==ind
            data[method].append(estim_prev[selected] - true_prev[selected])

    nmethods = len(method_names)
    boxwidth = binwidth/(nmethods+4)
    for i,bin in enumerate(bins):
        boxdata = [data[method][i] for method in method_names]
        positions = [bin+(i*boxwidth)+2*boxwidth for i,_ in enumerate(method_names)]
        box = boxplot(boxdata, showmeans=False, positions=positions, widths=boxwidth, sym='+', patch_artist=True)
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
        minor_xticks_labels.append(f'[{bins[i]:.2f}-{bins[i + 1]:.2f}' + (')' if i < len(bins)-2 else ']'))
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
    # ax.set_ylim(-1, 1)
    ax.set_xlim(0, 1)

    save_or_show(savepath)


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


def error_by_drift(method_names, true_prevs, estim_prevs, tr_prevs, n_bins=20, error_name='ae', show_std=True,
                   logscale=False,
                   title=f'Quantification error as a function of distribution shift',
                   savepath=None):

    fig, ax = plt.subplots()
    ax.grid()

    x_error = qp.error.ae
    y_error = getattr(qp.error, error_name)

    # join all data, and keep the order in which the methods appeared for the first time
    data = defaultdict(lambda:{'x':np.empty(shape=(0)), 'y':np.empty(shape=(0))})
    method_order = []
    for method, test_prevs_i, estim_prevs_i, tr_prev_i in zip(method_names, true_prevs, estim_prevs, tr_prevs):
        tr_prev_i = np.repeat(tr_prev_i.reshape(1,-1), repeats=test_prevs_i.shape[0], axis=0)

        tr_test_drifts = x_error(test_prevs_i, tr_prev_i)
        data[method]['x'] = np.concatenate([data[method]['x'], tr_test_drifts])

        method_drifts = y_error(test_prevs_i, estim_prevs_i)
        data[method]['y'] = np.concatenate([data[method]['y'], method_drifts])

        if method not in method_order:
            method_order.append(method)

    bins = np.linspace(0, 1, n_bins+1)
    binwidth = 1 / n_bins
    min_x, max_x = None, None
    for method in method_order:
        tr_test_drifts = data[method]['x']
        method_drifts = data[method]['y']
        if logscale:
            method_drifts=np.log(1+method_drifts)

        inds = np.digitize(tr_test_drifts, bins, right=True)
        xs, ys, ystds = [], [], []
        for ind in range(len(bins)):
            selected = inds==ind
            if selected.sum() > 0:
                xs.append(ind*binwidth)
                ys.append(np.mean(method_drifts[selected]))
                ystds.append(np.std(method_drifts[selected]))

        xs = np.asarray(xs)
        ys = np.asarray(ys)
        ystds = np.asarray(ystds)

        min_x_method, max_x_method = xs.min(), xs.max()
        min_x = min_x_method if min_x is None or min_x_method < min_x else min_x
        max_x = max_x_method if max_x is None or max_x_method > max_x else max_x

        ax.errorbar(xs, ys, fmt='-', marker='o', label=method, markersize=3, zorder=2)
        if show_std:
            ax.fill_between(xs, ys-ystds, ys+ystds, alpha=0.25)

    ax.set(xlabel=f'Distribution shift between training set and test sample',
           ylabel=f'{error_name.upper()} (true distribution, predicted distribution)',
           title=title)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim(min_x, max_x)

    save_or_show(savepath)


def save_or_show(savepath):
    # if savepath is specified, then saves the plot in that path; otherwise the plot is shown
    if savepath is not None:
        qp.util.create_parent_dir(savepath)
        # plt.tight_layout()
        plt.savefig(savepath, bbox_inches='tight')
    else:
        plt.show()

