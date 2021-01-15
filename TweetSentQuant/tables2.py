import quapy as qp
import numpy as np
from os import makedirs
# from evaluate import evaluate_directory, statistical_significance, get_ranks_from_Gao_Sebastiani
import sys, os
import pickle
from experiments import result_path
from result_manager import ResultSet
from tabular import Table

tables_path = './tables'
MAXTONE = 50  # sets the intensity of the maximum color reached by the worst (red) and best (green) results

makedirs(tables_path, exist_ok=True)

sample_size = 100
qp.environ['SAMPLE_SIZE'] = sample_size


nice = {
    'mae':'AE',
    'mrae':'RAE',
    'ae':'AE',
    'rae':'RAE',
    'svmkld': 'SVM(KLD)',
    'svmnkld': 'SVM(NKLD)',
    'svmq': 'SVM(Q)',
    'svmae': 'SVM(AE)',
    'svmnae': 'SVM(NAE)',
    'svmmae': 'SVM(AE)',
    'svmmrae': 'SVM(RAE)',
    'quanet': 'QuaNet',
    'hdy': 'HDy',
    'dys': 'DyS',
    'svmperf':'',
    'sanders': 'Sanders',
    'semeval13': 'SemEval13',
    'semeval14': 'SemEval14',
    'semeval15': 'SemEval15',
    'semeval16': 'SemEval16',
    'Average': 'Average'
}



def nicerm(key):
    return '\mathrm{'+nice[key]+'}'


def load_Gao_Sebastiani_previous_results():
    def rename(method):
        old2new = {
            'kld': 'svmkld',
            'nkld': 'svmnkld',
            'qbeta2': 'svmq',
            'em': 'sld'
        }
        return old2new.get(method, method)

    gao_seb_results = {}
    with open('./Gao_Sebastiani_results.txt', 'rt') as fin:
        lines = fin.readlines()
        for line in lines[1:]:
            line = line.strip()
            parts = line.lower().split()
            if len(parts) == 4:
                dataset, method, ae, rae = parts
            else:
                method, ae, rae = parts
            learner, method = method.split('-')
            method = rename(method)
            gao_seb_results[f'{dataset}-{method}-ae'] = float(ae)
            gao_seb_results[f'{dataset}-{method}-rae'] = float(rae)
    return gao_seb_results


def get_ranks_from_Gao_Sebastiani():
    gao_seb_results = load_Gao_Sebastiani_previous_results()
    datasets = set([key.split('-')[0] for key in gao_seb_results.keys()])
    methods = np.sort(np.unique([key.split('-')[1] for key in gao_seb_results.keys()]))
    ranks = {}
    for metric in ['ae', 'rae']:
        for dataset in datasets:
            scores = [gao_seb_results[f'{dataset}-{method}-{metric}'] for method in methods]
            order = np.argsort(scores)
            sorted_methods = methods[order]
            for i, method in enumerate(sorted_methods):
                ranks[f'{dataset}-{method}-{metric}'] = i+1
        for method in methods:
            rankave = np.mean([ranks[f'{dataset}-{method}-{metric}'] for dataset in datasets])
            ranks[f'Average-{method}-{metric}'] = rankave
    return ranks, gao_seb_results


def save_table(path, table):
    print(f'saving results in {path}')
    with open(path, 'wt') as foo:
        foo.write(table)



datasets = qp.datasets.TWITTER_SENTIMENT_DATASETS_TEST
evaluation_measures = [qp.error.ae, qp.error.rae]
gao_seb_methods = ['cc', 'acc', 'pcc', 'pacc', 'sld', 'svmq', 'svmkld', 'svmnkld']
new_methods = []

def addfunc(dataset, method, loss):
    path = result_path(dataset, method, 'm'+loss if not loss.startswith('m') else loss)
    if os.path.exists(path):
        true_prevs, estim_prevs, _, _, _, _ = pickle.load(open(path, 'rb'))
        err_fn = getattr(qp.error, loss)
        errors = err_fn(true_prevs, estim_prevs)
        return errors
    return None


gao_seb_ranks, gao_seb_results = get_ranks_from_Gao_Sebastiani()

for i, eval_func in enumerate(evaluation_measures):

    # Tables evaluation scores for AE and RAE (two tables)
    # ----------------------------------------------------

    eval_name = eval_func.__name__
    added_methods = ['svm' + eval_name] + new_methods
    methods = gao_seb_methods + added_methods
    nold_methods = len(gao_seb_methods)
    nnew_methods = len(added_methods)

    table = Table(rows=datasets, cols=methods, addfunc=addfunc)

    # fill table
    for dataset in datasets:
        for method in methods:
            table.add(dataset, method, eval_name)

    tabular = """
    \\begin{tabularx}{\\textwidth}{|c||""" + ('Y|'*len(gao_seb_methods))+ '|' + ('Y|'*len(added_methods)) + """} \hline
      & \multicolumn{"""+str(nold_methods)+"""}{c||}{Methods tested in~\cite{Gao:2016uq}} & \multicolumn{"""+str(nnew_methods)+"""}{c|}{} \\\\ \hline
    """

    rowreplace={dataset: nice.get(dataset, dataset.upper()) for dataset in datasets}
    colreplace={method:'\side{' + nice.get(method, method.upper()) +'$^{' + nicerm(eval_name) + '}$} ' for method in methods}

    tabular += table.latextabular(rowreplace=rowreplace, colreplace=colreplace)
    tabular += "\n\end{tabularx}"

    save_table(f'./tables/tab_results_{eval_name}.new2.tex', tabular)



    # Tables ranks for AE and RAE (two tables)
    # ----------------------------------------------------
    def addfuncRank(dataset, method):
        rank = table.get(dataset, method, 'rank')
        if rank is None:
            return None
        return [rank]

    methods = gao_seb_methods
    nold_methods = len(gao_seb_methods)

    ranktable = Table(rows=datasets, cols=methods, addfunc=addfuncRank)
    # fill table
    for dataset in datasets:
        for method in methods:
            ranktable.add(dataset, method)


    tabular = """
    \\begin{tabularx}{\\textwidth}{|c||""" + ('Y|' * len(gao_seb_methods)) + """} \hline
          & \multicolumn{""" + str(nold_methods) + """}{c||}{Methods tested in~\cite{Gao:2016uq}}  \\\\ \hline
    """
    for method in methods:
        tabular += ' & \side{' + nice.get(method, method.upper()) +'$^{' + nicerm(eval_name) + '}$} '
    tabular += '\\\\\hline\n'

    for dataset in datasets:
        tabular += nice.get(dataset, dataset.upper()) + ' '
        for method in methods:
            newrank = ranktable.get(dataset, method)
            oldrank = gao_seb_ranks[f'{dataset}-{method}-{eval_name}']
            if newrank is None:
                newrank = '--'
            else:
                newrank =  f'{int(newrank)}'
            tabular += ' & ' + f'{newrank}' + f' ({oldrank}) ' + ranktable.get_color(dataset, method)
        tabular += '\\\\\hline\n'

    tabular += 'Average & '
    for method in methods:
        newrank = ranktable.get_col_average(method)
        oldrank = gao_seb_ranks[f'Average-{method}-{eval_name}']
        if newrank is None or np.isnan(newrank):
            newrank = '--'
        else:
            newrank = f'{newrank:.1f}'
        oldrank = f'{oldrank:.1f}'
        tabular += ' & ' + f'{newrank}' + f' ({oldrank}) ' + ranktable.get_color(dataset, method)
    tabular += '\\\\\hline\n'

    tabular += "\end{tabularx}"

    save_table(f'./tables/tab_rank_{eval_name}.new2.tex', tabular)


print("[Done]")