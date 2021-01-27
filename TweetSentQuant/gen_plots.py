import quapy as qp
import settings
import os
import pathlib
import pickle
from glob import glob
import sys
from TweetSentQuant.util import nicename
from os.path import join


qp.environ['SAMPLE_SIZE'] = settings.SAMPLE_SIZE
plotext='png'

resultdir = './results'
plotdir = './plots'
os.makedirs(plotdir, exist_ok=True)

def gather_results(methods, error_name):
    method_names, true_prevs, estim_prevs, tr_prevs = [], [], [], []
    for method in methods:
        for experiment in glob(f'{resultdir}/*-{method}-m{error_name}.pkl'):
            true_prevalences, estim_prevalences, tr_prev, te_prev, te_prev_estim, best_params = pickle.load(open(experiment, 'rb'))
            method_names.append(nicename(method))
            true_prevs.append(true_prevalences)
            estim_prevs.append(estim_prevalences)
            tr_prevs.append(tr_prev)
    return method_names, true_prevs, estim_prevs, tr_prevs


def plot_error_by_drift(methods, error_name, logscale=False, path=None):
    print('plotting error by drift')
    if path is not None:
        path = join(path, f'error_by_drift_{error_name}.{plotext}')
    method_names, true_prevs, estim_prevs, tr_prevs = gather_results(methods, error_name)
    qp.plot.error_by_drift(
        method_names,
        true_prevs,
        estim_prevs,
        tr_prevs,
        n_bins=20,
        error_name=error_name,
        show_std=False,
        logscale=logscale,
        title=f'Quantification error as a function of distribution shift',
        savepath=path
    )


def diagonal_plot(methods, error_name, path=None):
    print('plotting diagonal plots')
    if path is not None:
        path = join(path, f'diag_{error_name}')
    method_names, true_prevs, estim_prevs, tr_prevs = gather_results(methods, error_name)
    qp.plot.binary_diagonal(method_names, true_prevs, estim_prevs, pos_class=0, title='Negative', legend=False, show_std=False, savepath=f'{path}_neg.{plotext}')
    qp.plot.binary_diagonal(method_names, true_prevs, estim_prevs, pos_class=1, title='Neutral',  legend=False, show_std=False, savepath=f'{path}_neu.{plotext}')
    qp.plot.binary_diagonal(method_names, true_prevs, estim_prevs, pos_class=2, title='Positive', legend=True, show_std=False, savepath=f'{path}_pos.{plotext}')


def binary_bias_global(methods, error_name, path=None):
    print('plotting bias global')
    if path is not None:
        path = join(path, f'globalbias_{error_name}')
    method_names, true_prevs, estim_prevs, tr_prevs = gather_results(methods, error_name)
    qp.plot.binary_bias_global(method_names, true_prevs, estim_prevs, pos_class=0, title='Negative', savepath=f'{path}_neg.{plotext}')
    qp.plot.binary_bias_global(method_names, true_prevs, estim_prevs, pos_class=1, title='Neutral', savepath=f'{path}_neu.{plotext}')
    qp.plot.binary_bias_global(method_names, true_prevs, estim_prevs, pos_class=2, title='Positive', savepath=f'{path}_pos.{plotext}')


def binary_bias_bins(methods, error_name, path=None):
    print('plotting bias local')
    if path is not None:
        path = join(path, f'localbias_{error_name}')
    method_names, true_prevs, estim_prevs, tr_prevs = gather_results(methods, error_name)
    qp.plot.binary_bias_bins(method_names, true_prevs, estim_prevs, pos_class=0, title='Negative', legend=False, savepath=f'{path}_neg.{plotext}')
    qp.plot.binary_bias_bins(method_names, true_prevs, estim_prevs, pos_class=1, title='Neutral', legend=False, savepath=f'{path}_neu.{plotext}')
    qp.plot.binary_bias_bins(method_names, true_prevs, estim_prevs, pos_class=2, title='Positive', legend=True, savepath=f'{path}_pos.{plotext}')


gao_seb_methods = ['cc', 'acc', 'pcc', 'pacc', 'sld', 'svmq', 'svmkld', 'svmnkld']
new_methods_ae = ['svmmae' , 'epaccmaeptr', 'epaccmaemae', 'hdy', 'quanet']
new_methods_rae = ['svmmrae' , 'epaccmraeptr', 'epaccmraemrae', 'hdy', 'quanet']

plot_error_by_drift(gao_seb_methods+new_methods_ae, error_name='ae', path=plotdir)
plot_error_by_drift(gao_seb_methods+new_methods_rae, error_name='rae', logscale=True, path=plotdir)

diagonal_plot(gao_seb_methods+new_methods_ae, error_name='ae', path=plotdir)
diagonal_plot(gao_seb_methods+new_methods_rae, error_name='rae', path=plotdir)

binary_bias_global(gao_seb_methods+new_methods_ae, error_name='ae', path=plotdir)
binary_bias_global(gao_seb_methods+new_methods_rae, error_name='rae', path=plotdir)

#binary_bias_bins(gao_seb_methods+new_methods_ae, error_name='ae', path=plotdir)
#binary_bias_bins(gao_seb_methods+new_methods_rae, error_name='rae', path=plotdir)

