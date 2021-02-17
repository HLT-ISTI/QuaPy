import quapy as qp
import numpy as np
from os import makedirs
import sys, os
import pickle
from experiments import result_path
from gen_tables import save_table, experiment_errors
from tabular import Table
import argparse

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
    'hdysld': 'HDy-SLD',
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate tables for Tweeter Sentiment Quantification')
    parser.add_argument('results', metavar='RESULT_PATH', type=str,
                        help='path to the directory containing the results of the methods tested in Gao & Sebastiani')
    parser.add_argument('newresults', metavar='RESULT_PATH', type=str,
                        help='path to the directory containing the results for the experimental methods')
    args = parser.parse_args()

    datasets = qp.datasets.TWITTER_SENTIMENT_DATASETS_TEST
    evaluation_measures = [qp.error.ae, qp.error.rae]
    gao_seb_methods = ['cc', 'acc', 'pcc', 'pacc', 'sld', 'svmq', 'svmkld', 'svmnkld']
    new_methods = ['hdy']  # methods added to the Gao & Sebastiani methods
    experimental_methods = ['hdysld']  # experimental

    for i, eval_func in enumerate(evaluation_measures):

        # Tables evaluation scores for AE and RAE (two tables)
        # ----------------------------------------------------

        eval_name = eval_func.__name__

        added_methods = ['svmm' + eval_name] + new_methods
        methods = gao_seb_methods + added_methods + experimental_methods
        nold_methods = len(gao_seb_methods)
        nnew_methods = len(added_methods)
        nexp_methods = len(experimental_methods)

        # fill data table
        table = Table(benchmarks=datasets, methods=methods)
        for dataset in datasets:
            for method in methods:
                if method in experimental_methods:
                    path = args.newresults
                else:
                    path = args.results
                table.add(dataset, method, experiment_errors(path, dataset, method, eval_name))

        # write the latex table
        tabular = """
        \\begin{tabularx}{\\textwidth}{|c||""" + ('Y|'*nold_methods) + '|' + ('Y|'*nnew_methods) + '|' + ('Y|'*nexp_methods) + """} \hline
          & \multicolumn{"""+str(nold_methods)+"""}{c||}{Methods tested in~\cite{Gao:2016uq}} & 
            \multicolumn{"""+str(nnew_methods)+"""}{c|}{} & 
            \multicolumn{"""+str(nexp_methods)+"""}{c|}{}\\\\ \hline
        """
        rowreplace={dataset: nice.get(dataset, dataset.upper()) for dataset in datasets}
        colreplace={method:'\side{' + nice.get(method, method.upper()) +'$^{' + nicerm(eval_name) + '}$} ' for method in methods}

        tabular += table.latexTabular(benchmark_replace=rowreplace, method_replace=colreplace)
        tabular += "\n\end{tabularx}"

        save_table(f'./tables/tab_results_{eval_name}.new.tex', tabular)

        # Tables ranks for AE and RAE (two tables)
        # ----------------------------------------------------
        # fill the data table
        ranktable = Table(benchmarks=datasets, methods=methods, missing='--')
        for dataset in datasets:
            for method in methods:
                ranktable.add(dataset, method, values=table.get(dataset, method, 'rank'))

        # write the latex table
        tabular = """
        \\begin{tabularx}{\\textwidth}{|c||""" + ('Y|'*nold_methods) + '|' + ('Y|'*nnew_methods) + '|' + ('Y|'*nexp_methods) + """} \hline
              & \multicolumn{"""+str(nold_methods)+"""}{c||}{Methods tested in~\cite{Gao:2016uq}} & 
            \multicolumn{"""+str(nnew_methods)+"""}{c|}{} & 
            \multicolumn{"""+str(nexp_methods)+"""}{c|}{}\\\\ \hline
        """
        for method in methods:
            tabular += ' & \side{' + nice.get(method, method.upper()) +'$^{' + nicerm(eval_name) + '}$} '
        tabular += '\\\\\hline\n'

        for dataset in datasets:
            tabular += nice.get(dataset, dataset.upper()) + ' '
            for method in methods:
                newrank = ranktable.get(dataset, method)
                if newrank != '--':
                    newrank = f'{int(newrank)}'
                color = ranktable.get_color(dataset, method)
                if color == '--':
                    color = ''
                tabular += ' & ' + f'{newrank}' + color
            tabular += '\\\\\hline\n'
        tabular += '\hline\n'

        tabular += 'Average '
        for method in methods:
            newrank = ranktable.get_average(method)
            if newrank != '--':
                newrank = f'{newrank:.1f}'
            color = ranktable.get_average(method, 'color')
            if color == '--':
                color = ''
            tabular += ' & ' + f'{newrank}' + color
        tabular += '\\\\\hline\n'
        tabular += "\end{tabularx}"

        save_table(f'./tables/tab_rank_{eval_name}.new.tex', tabular)

    print("[Done]")
