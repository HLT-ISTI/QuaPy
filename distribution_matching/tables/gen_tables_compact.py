from distribution_matching.commons import (ADJUSTMENT_METHODS, BIN_METHODS, DISTR_MATCH_METHODS, MAX_LIKE_METHODS,
                                           METHODS, FULL_METHOD_LIST, BIN_ADJUSTMENT_METHODS)
import quapy as qp
from os import makedirs
import os

from tabular import Table
import pandas as pd

tables_path = '.'
# makedirs(tables_path, exist_ok=True)

MAXTONE = 35  # sets the intensity of the maximum color reached by the worst (red) and best (green) results
SHOW_STD = False

NUM_ADJUSTMENT_METHODS = len(ADJUSTMENT_METHODS)
NUM_BIN_ADJUSTMENT_METHODS = len(BIN_ADJUSTMENT_METHODS)
NUM_MAXIMUM_LIKELIHOOD_METHODS = len(MAX_LIKE_METHODS)
NUM_DISTRIBUTION_MATCHING_METHODS = len(DISTR_MATCH_METHODS)

qp.environ['SAMPLE_SIZE'] = 100

nice_bench = {
    'sanders': 'Sanders',
    'semeval13': 'SemEval13',
    'semeval14': 'SemEval14',
    'semeval15': 'SemEval15',
    'semeval16': 'SemEval16',
}

nice_method = {
    'EDy+': 'EDy',
    'ACC+': 'ACC',
    'PACC+': 'PACC'
}


def save_table(path, table):
    print(f'saving results in {path}')
    with open(path, 'wt') as foo:
        foo.write(table)

def new_table(datasets, methods):
    return Table(
        benchmarks=datasets,
        methods=methods,
        ttest='wilcoxon',
        prec_mean=4,
        show_std=SHOW_STD,
        prec_std=4,
        clean_zero=(eval=='mae'),
        average=False,
        maxtone=MAXTONE
    )


def make_table(tabs, eval, benchmark_groups, benchmark_names, compact=False, binary=False):

    num_adjustment = NUM_BIN_ADJUSTMENT_METHODS if binary else NUM_ADJUSTMENT_METHODS
    n_methods = len(BIN_METHODS) if binary else len(METHODS)
    assert n_methods == (num_adjustment+NUM_DISTRIBUTION_MATCHING_METHODS+NUM_MAXIMUM_LIKELIHOOD_METHODS), \
        "Unexpected number of methods"

    cline = "\cline{2-" + str(n_methods+ 1) + "}"

    # write the latex table
    tabular = """
            \\begin{tabular}{|c|""" + ('c|' * num_adjustment) + ('c|' * NUM_DISTRIBUTION_MATCHING_METHODS) +  ('c|' * NUM_MAXIMUM_LIKELIHOOD_METHODS) + """} """ + cline + """           
            \multicolumn{1}{c}{} & 
            \multicolumn{"""+str(num_adjustment)+"""}{|c}{Adjustment} & 
            \multicolumn{"""+str(NUM_DISTRIBUTION_MATCHING_METHODS)+"""}{|c|}{Distribution Matching} & 
            \multicolumn{"""+str(NUM_MAXIMUM_LIKELIHOOD_METHODS)+"""}{c|}{Maximum Likelihood} \\\\
            \hline               
            """
    for i, (tab, group, name) in enumerate(zip(tabs, benchmark_groups, benchmark_names)):
        tablines = tab.latexTabular(benchmark_replace=nice_bench, method_replace=nice_method, endl='\\\\'+ cline, aslines=True)
        tablines[0] = tablines[0].replace('\multicolumn{1}{c|}{}', '\\textbf{'+name+'}')
        if compact or len(tab.benchmarks)==1:
            # if compact, keep the method names and the average; discard the rest
            tabular += tablines[0] + '\n' + tablines[1 if len(tab.benchmarks)==1 else -1] + '\n'
        else:
            tabular += '\n'.join(tablines)

        # tabular += "\n" + "\\textit{Rank} & " + tab.getRankTable(prec_mean=0 if name.startswith('LeQua') else 1).latexAverage()
        # if i < (len(tabs) - 1):
        tabular += "\\hline\n"
        # else:
        #     tabular += "\n"
    tabular += "\end{tabular}"
    return tabular


def gen_tables_uci_multiclass(eval):

    print('Generating table for UCI Multiclass Datasets', eval)
    dir_results = f'../results/ucimulti/{eval}'

    datasets = qp.datasets.UCI_MULTICLASS_DATASETS

    tab =  new_table(datasets, METHODS)

    for dataset in datasets:
        print(f'\t Dataset: {dataset}: ', end='')
        for method in METHODS:
            result_path = f'{dir_results}/{method}_{dataset}.dataframe'
            if os.path.exists(result_path):
                df = pd.read_csv(result_path)
                print(f'{method}', end=' ')
                tab.add(dataset, method, df[eval].values)
            else:
                print(f'MISSING-{method}', end=' ')
        print()

    return tab


def gen_tables_uci_bin(eval):

    print('Generating table for UCI Datasets', eval)
    dir_results = f'../results/binary/{eval}'

    exclude = ['acute.a', 'acute.b', 'iris.1', 'balance.2']
    datasets = [x for x in qp.datasets.UCI_BINARY_DATASETS if x not in exclude]

    tab =  new_table(datasets, BIN_METHODS)

    for dataset in datasets:
        print(f'\t Dataset: {dataset}: ', end='')
        for method in BIN_METHODS:
            result_path = f'{dir_results}/{method}_{dataset}.dataframe'
            if os.path.exists(result_path):
                df = pd.read_csv(result_path)
                print(f'{method}', end=' ')
                tab.add(dataset, method, df[eval].values)
            else:
                print(f'MISSING-{method}', end=' ')

    return tab

    

def gen_tables_tweet(eval):

    print('Generating table for Twitter', eval)
    dir_results = f'../results/tweet/{eval}'

    datasets = qp.datasets.TWITTER_SENTIMENT_DATASETS_TEST

    tab =  new_table(datasets, METHODS)

    for dataset in datasets:
        print(f'\t Dataset: {dataset}: ', end='')
        for method in METHODS:
            result_path = f'{dir_results}/{method}_{dataset}.dataframe'
            if os.path.exists(result_path):
                df = pd.read_csv(result_path)
                print(f'{method}', end=' ')
                tab.add(dataset, method, df[eval].values)
            else:
                print(f'MISSING-{method}', end=' ')
        print()

    return tab


def gen_tables_lequa(Methods, task, eval):
    # generating table for LeQua-T1A or Lequa-T1B; only one table with two rows, one for MAE, another for MRAE

    tab = new_table([task], Methods)

    print('Generating table for T1A@Lequa', eval, end='')
    dir_results = f'../results/lequa/{task}/{eval}'

    for method in Methods:
        result_path = f'{dir_results}/{method}.dataframe'
        if os.path.exists(result_path):
            df = pd.read_csv(result_path)
            print(f'{method}', end=' ')
            tab.add(task, method, df[eval].values)
        else:
            print(f'MISSING-{method}', end=' ')
    print()

    return tab


if __name__ == '__main__':

    os.makedirs('./latex', exist_ok=True)

    for eval in ['mae', 'mrae']:
        tabs = []
        tabs.append(gen_tables_tweet(eval))
        tabs.append(gen_tables_uci_multiclass(eval))
        tabs.append(gen_tables_lequa(METHODS, 'T1B', eval))

        names = ['Tweets', 'UCI-multi', 'LeQua']
        table = make_table(tabs, eval, benchmark_groups=tabs, benchmark_names=names)
        save_table(f'./latex/multiclass_{eval}.tex', table)

    for eval in ['mae', 'mrae']:
        tabs = []
        tabs.append(gen_tables_uci_bin(eval))
        
        # print uci-binary with all datasets for the appendix
        table = make_table(tabs, eval, benchmark_groups=tabs, benchmark_names=['UCI-binary'], binary=True)
        save_table(f'./latex/ucibinary_{eval}.tex', table)
        
        # print uci-bin compacted plus lequa-T1A for the main body
        tabs.append(gen_tables_lequa(BIN_METHODS, 'T1A', eval))
        table = make_table(tabs, eval, benchmark_groups=tabs, benchmark_names=['UCI-binary', 'LeQua'], compact=True, binary=True)
        save_table(f'./latex/binary_{eval}.tex', table)

    print("[Tables Done] runing latex")
    os.chdir('./latex/')
    os.system('pdflatex tables_compact.tex')
    os.system('rm tables_compact.aux tables_compact.bbl tables_compact.blg tables_compact.log tables_compact.out tables_compact.dvi')

