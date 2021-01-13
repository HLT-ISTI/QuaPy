import quapy as qp
import numpy as np
from os import makedirs
# from evaluate import evaluate_directory, statistical_significance, get_ranks_from_Gao_Sebastiani
import sys, os
import pickle
from experiments import result_path
from result_manager import ResultSet


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

def color_from_rel_rank(rel_rank, maxtone=100):
    rel_rank = rel_rank*2-1
    if rel_rank < 0:
        color = 'red'
        tone = maxtone*(-rel_rank)
    else:
        color = 'green'
        tone = maxtone*rel_rank
    return '\cellcolor{' + color + f'!{int(tone)}' + '}'

def color_from_abs_rank(abs_rank, n_methods, maxtone=100):
    rel_rank = 1.-(abs_rank-1.)/(n_methods-1)
    return color_from_rel_rank(rel_rank, maxtone)


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


# Tables evaluation scores for AE and RAE (two tables)
# ----------------------------------------------------

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
        return {
            'values': errors,
        }
    return None

def addave(method, tables):
    values = []
    for table in tables:
        mean = table.get(method, 'values', missing=None)
        if mean is None:
            return None
        values.append(mean)
    values = np.concatenate(values)
    return {
        'values': values
    }

def addrankave(method, tables):
    values = []
    for table in tables:
        rank = table.get(method, 'rank', missing=None)
        if rank is None:
            return None
        values.append(rank)
    return {
        'values': np.asarray(values)
    }


TABLES = {eval_func.__name__:{} for eval_func in evaluation_measures}

for i, eval_func in enumerate(evaluation_measures):
    eval_name = eval_func.__name__
    added_methods = ['svm' + eval_name] + new_methods
    methods = gao_seb_methods + added_methods
    nold_methods = len(gao_seb_methods)
    nnew_methods = len(added_methods)

    # fill table
    TABLE = TABLES[eval_name]
    for dataset in datasets:
        TABLE[dataset] = ResultSet(dataset, addfunc, show_std=False, test="ttest_ind_from_stats")
        for method in methods:
            TABLE[dataset].add(method, dataset, method, eval_name)

    TABLE['Average'] = ResultSet('ave', addave, show_std=False, test="ttest_ind_from_stats")
    for method in methods:
        TABLE['Average'].add(method, method, [TABLE[dataset] for dataset in datasets])

    tabular = """
    \\begin{tabularx}{\\textwidth}{|c||""" + ('Y|'*len(gao_seb_methods))+ '|' + ('Y|'*len(added_methods)) + """} \hline
      & \multicolumn{"""+str(nold_methods)+"""}{c||}{Methods tested in~\cite{Gao:2016uq}} & \multicolumn{"""+str(nnew_methods)+"""}{c|}{} \\\\ \hline
    """

    for method in methods:
        tabular += ' & \side{' + nice.get(method, method.upper()) +'$^{' + nicerm(eval_name) + '}$} '
    tabular += '\\\\\hline\n'

    for dataset in datasets + ['Average']:
        if dataset == 'Average': tabular+= '\line\n'
        tabular += nice.get(dataset, dataset.upper()) + ' '
        for method in methods:
            tabular += ' & ' + TABLE[dataset].latex(method)
        tabular += '\\\\\hline\n'

    tabular += "\end{tabularx}"

    save_table(f'./tables/tab_results_{eval_name}.new.tex', tabular)


gao_seb_ranks, gao_seb_results = get_ranks_from_Gao_Sebastiani()

# Tables ranks for AE and RAE (two tables)
# ----------------------------------------------------
for i, eval_func in enumerate(evaluation_measures):
    eval_name = eval_func.__name__
    methods = gao_seb_methods
    nold_methods = len(gao_seb_methods)

    TABLE = TABLES[eval_name]
    TABLE['Average'] = ResultSet('ave', addrankave, show_std=False, test="ttest_ind_from_stats")
    for method in methods:
        TABLE['Average'].add(method, method, [TABLE[dataset] for dataset in datasets])


    tabular = """
    \\begin{tabularx}{\\textwidth}{|c||""" + ('Y|' * len(gao_seb_methods)) + """} \hline
          & \multicolumn{""" + str(nold_methods) + """}{c||}{Methods tested in~\cite{Gao:2016uq}}  \\\\ \hline
    """

    for method in methods:
        tabular += ' & \side{' + nice.get(method, method.upper()) +'$^{' + nicerm(eval_name) + '}$} '
    tabular += '\\\\\hline\n'

    for dataset in datasets + ['Average']:
        if dataset == 'Average':
            tabular += '\line\n'
        else:
            TABLE[dataset].change_compare('rank')
        tabular += nice.get(dataset, dataset.upper()) + ' '
        for method in gao_seb_methods:
            if dataset == 'Average':
                method_rank = TABLE[dataset].get(method, 'mean')
            else:
                method_rank = TABLE[dataset].get(method, 'rank')
            gao_seb_rank = gao_seb_ranks[f'{dataset}-{method}-{eval_name}']
            if dataset == 'Average':
                if method_rank != '--':
                    method_rank = f'{method_rank:.1f}'
                gao_seb_rank = f'{gao_seb_rank:.1f}'
            tabular += ' & ' + f'{method_rank}' + f' ({gao_seb_rank}) ' + TABLE[dataset].get_color(method)
        tabular += '\\\\\hline\n'
    tabular += "\end{tabularx}"

    save_table(f'./tables/tab_rank_{eval_name}.new.tex', tabular)


print("[Done]")