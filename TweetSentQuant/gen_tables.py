import quapy as qp
import numpy as np
from os import makedirs
import sys, os
import pickle
import argparse
from TweetSentQuant.util import nicename, get_ranks_from_Gao_Sebastiani
import settings
from experiments import result_path
from tabular import Table

tables_path = './tables'
MAXTONE = 50  # sets the intensity of the maximum color reached by the worst (red) and best (green) results

makedirs(tables_path, exist_ok=True)

qp.environ['SAMPLE_SIZE'] = settings.SAMPLE_SIZE


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
    new_methods = ['hdy', 'quanet']

    gao_seb_ranks, gao_seb_results = get_ranks_from_Gao_Sebastiani()

    for i, eval_func in enumerate(evaluation_measures):

        # Tables evaluation scores for AE and RAE (two tables)
        # ----------------------------------------------------

        eval_name = eval_func.__name__
        added_methods = ['svmm' + eval_name, f'epaccm{eval_name}ptr', f'epaccm{eval_name}m{eval_name}'] + new_methods
        methods = gao_seb_methods + added_methods
        nold_methods = len(gao_seb_methods)
        nnew_methods = len(added_methods)

        # fill data table
        table = Table(benchmarks=datasets, methods=methods)
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
        rowreplace={dataset: nicename(dataset) for dataset in datasets}
        colreplace={method: nicename(method, eval_name, side=True) for method in methods}

        tabular += table.latexTabular(benchmark_replace=rowreplace, method_replace=colreplace)
        tabular += """
            \end{tabular}%
            }
        """

        save_table(f'./tables/tab_results_{eval_name}.new.tex', tabular)

        # Tables ranks for AE and RAE (two tables)
        # ----------------------------------------------------
        methods = gao_seb_methods

        table.dropMethods(added_methods)

        # fill the data table
        ranktable = Table(benchmarks=datasets, methods=methods, missing='--')
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
            tabular += ' & ' + nicename(method, eval_name, side=True)
        tabular += "\\\\\hline\n"

        for dataset in datasets:
            tabular += nicename(dataset) + ' '
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
