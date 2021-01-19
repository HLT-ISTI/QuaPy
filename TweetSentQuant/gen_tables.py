import quapy as qp
import numpy as np
from os import makedirs
import sys, os
import pickle
import argparse

import settings
from experiments import result_path
from tabular import Table

tables_path = './tables'
MAXTONE = 50  # sets the intensity of the maximum color reached by the worst (red) and best (green) results

makedirs(tables_path, exist_ok=True)

qp.environ['SAMPLE_SIZE'] = settings.SAMPLE_SIZE


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


def experiment_errors(path, dataset, method, loss):
    path = result_path(path, dataset, method, 'm'+loss if not loss.startswith('m') else loss)
    if os.path.exists(path):
        true_prevs, estim_prevs, _, _, _, _ = pickle.load(open(path, 'rb'))
        err_fn = getattr(qp.error, loss)
        errors = err_fn(true_prevs, estim_prevs)
        return errors
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate tables for Tweeter Sentiment Quantification')
    parser.add_argument('results', metavar='RESULT_PATH', type=str,
                        help='path to the directory where to store the results')
    args = parser.parse_args()

    datasets = qp.datasets.TWITTER_SENTIMENT_DATASETS_TEST
    evaluation_measures = [qp.error.ae, qp.error.rae]
    gao_seb_methods = ['cc', 'acc', 'pcc', 'pacc', 'sld', 'svmq', 'svmkld', 'svmnkld']
    new_methods = ['hdy']

    gao_seb_ranks, gao_seb_results = get_ranks_from_Gao_Sebastiani()

    for i, eval_func in enumerate(evaluation_measures):

        # Tables evaluation scores for AE and RAE (two tables)
        # ----------------------------------------------------

        eval_name = eval_func.__name__
        added_methods = ['svmm' + eval_name] + new_methods
        methods = gao_seb_methods + added_methods
        nold_methods = len(gao_seb_methods)
        nnew_methods = len(added_methods)

        # fill data table
        table = Table(rows=datasets, cols=methods)
        for dataset in datasets:
            for method in methods:
                table.add(dataset, method, experiment_errors(args.results, dataset, method, eval_name))

        # write the latex table
        # tabular = """
        # \\begin{tabularx}{\\textwidth}{|c||""" + ('Y|'*nold_methods)+ '|' + ('Y|'*nnew_methods) + """} \hline
        #   & \multicolumn{"""+str(nold_methods)+"""}{c||}{Methods tested in~\cite{Gao:2016uq}} &
        #     \multicolumn{"""+str(nnew_methods)+"""}{c|}{} \\\\ \hline
        # """
        tabular = """
        \\resizebox{\\textwidth}{!}{%
                \\begin{tabular}{|c||""" + ('c|' * nold_methods) + '|' + ('c|' * nnew_methods) + """} \hline
                  & \multicolumn{""" + str(nold_methods) + """}{c||}{Methods tested in~\cite{Gao:2016uq}} & 
                    \multicolumn{""" + str(nnew_methods) + """}{c|}{} \\\\ \hline
                """
        rowreplace={dataset: nice.get(dataset, dataset.upper()) for dataset in datasets}
        colreplace={method:'\side{' + nice.get(method, method.upper()) +'$^{' + nicerm(eval_name) + '}$} ' for method in methods}

        tabular += table.latexTabular(rowreplace=rowreplace, colreplace=colreplace)
        tabular += """
            \end{tabular}%
            }
        """

        save_table(f'./tables/tab_results_{eval_name}.new.tex', tabular)

        # Tables ranks for AE and RAE (two tables)
        # ----------------------------------------------------
        methods = gao_seb_methods

        # fill the data table
        ranktable = Table(rows=datasets, cols=methods, missing='--')
        for dataset in datasets:
            for method in methods:
                ranktable.add(dataset, method, values=table.get(dataset, method, 'rank'))

        # write the latex table
        tabular = """
        \\resizebox{\\textwidth}{!}{%
        \\begin{tabular}{|c||""" + ('c|' * len(gao_seb_methods)) + """} \hline
              & \multicolumn{""" + str(nold_methods) + """}{c|}{Methods tested in~\cite{Gao:2016uq}}  \\\\ \hline
        """
        for method in methods:
            tabular += ' & \side{' + nice.get(method, method.upper()) +'$^{' + nicerm(eval_name) + '}$} '
        tabular += "\\\\\hline\n"

        for dataset in datasets:
            tabular += nice.get(dataset, dataset.upper()) + ' '
            for method in methods:
                newrank = ranktable.get(dataset, method)
                oldrank = gao_seb_ranks[f'{dataset}-{method}-{eval_name}']
                if newrank != '--':
                    newrank = f'{int(newrank)}'
                color = ranktable.get_color(dataset, method)
                if color == '--':
                    color = ''
                tabular += ' & ' + f'{newrank}' + f' ({oldrank}) ' + color
            tabular += '\\\\\hline\n'
        tabular += '\hline\n'

        tabular += 'Average '
        for method in methods:
            newrank = ranktable.get_average(method)
            oldrank = gao_seb_ranks[f'Average-{method}-{eval_name}']
            if newrank != '--':
                newrank = f'{newrank:.1f}'
            oldrank = f'{oldrank:.1f}'
            color = ranktable.get_average(method, 'color')
            if color == '--':
                color = ''
            tabular += ' & ' + f'{newrank}' + f' ({oldrank}) ' + color
        tabular += '\\\\\hline\n'
        tabular += """
        \end{tabular}%
        }
        """

        save_table(f'./tables/tab_rank_{eval_name}.new.tex', tabular)

    print("[Done]")
