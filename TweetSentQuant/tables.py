import quapy as qp
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



# results_dict = evaluate_directory('results/*.pkl', evaluation_measures)
# stats = {
#     dataset : {
#         'mae': statistical_significance(f'results/{dataset}-*-mae-run?.pkl', ae),
#         'mrae': statistical_significance(f'results/{dataset}-*-mrae-run?.pkl', rae),
#     } for dataset in datasets
# }

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
    'semeval16': 'SemEval16'
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


def save_table(path, table):
    print(f'saving results in {path}')
    with open(path, 'wt') as foo:
        foo.write(table)


# Tables evaluation scores for AE and RAE (two tables)
# ----------------------------------------------------



datasets = qp.datasets.TWITTER_SENTIMENT_DATASETS_TEST
evaluation_measures = [qp.error.ae, qp.error.rae]
gao_seb_methods = ['cc', 'acc', 'pcc', 'pacc', 'emq', 'svmq', 'svmkld', 'svmnkld']

results_dict = {}
stats={}
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


for i, eval_func in enumerate(evaluation_measures):
    eval_name = eval_func.__name__
    added_methods = ['svm' + eval_name]  # , 'quanet', 'dys']
    methods = gao_seb_methods + added_methods
    nold_methods = len(gao_seb_methods)
    nnew_methods = len(added_methods)

    # fill table
    TABLE = {}
    for dataset in datasets:
        TABLE[dataset] = ResultSet(dataset, addfunc, show_std=False, test="ttest_ind_from_stats", maxtone=50,
                                   remove_mean='0.' if eval_func == qp.error.ae else '')
        for method in methods:
            TABLE[dataset].add(method, dataset, method, eval_name)

    tabular = """
    \\begin{tabularx}{\\textwidth}{|c||""" + ('Y|'*len(gao_seb_methods))+ '|' + ('Y|'*len(added_methods)) + """} \hline
      & \multicolumn{"""+str(nold_methods)+"""}{c||}{Methods tested in~\cite{Gao:2016uq}} & \multicolumn{"""+str(nnew_methods)+"""}{c||}{} \\\\ \hline
    """

    for method in methods:
        tabular += ' & \side{' + nice.get(method, method.upper()) +'$^{' + nicerm(eval_name) + '}$} '
    tabular += '\\\\\hline\n'

    for dataset in datasets:
        tabular += nice.get(dataset, dataset.upper()) + ' '
        for method in methods:
            tabular += ' & ' + TABLE[dataset].latex(method)
        tabular += '\\\\\hline\n'
    tabular += "\end{tabularx}"

    save_table(f'./tables/tab_results_{eval_name}.new.tex', tabular)

sys.exit(0)

# gao_seb_ranks, gao_seb_results = get_ranks_from_Gao_Sebastiani()

# Tables ranks for AE and RAE (two tables)
# ----------------------------------------------------
# for i, eval_func in enumerate(evaluation_measures):
#     eval_name = eval_func.__name__
#     methods = ['cc', 'acc', 'pcc', 'pacc', 'emq', 'svmq', 'svmkld', 'svmnkld']
#     table = """
#     \\begin{table}[h]
#     """
#     if i == 0:
#         caption = """
#           \caption{Rank positions of the quantification methods in the AE
#           experiments, and (between parentheses) the rank positions
#           obtained in the evaluation of~\cite{Gao:2016uq}.}
#         """
#     else:
#         caption = "\caption{Same as Table~\\ref{tab:maeranks}, but with " + nice[eval_name] + " instead of AE.}"
#     table += caption + """
#             \\begin{center}
#             \\resizebox{\\textwidth}{!}{
#         """
#     tabular = """
#         \\begin{tabularx}{\\textwidth}{|c||Y|Y|Y|Y|Y|Y|Y|Y|} \hline
#           & \multicolumn{8}{c|}{Methods tested in~\cite{Gao:2016uq}}  \\\\ \hline
#     """
#
#     for method in methods:
#         tabular += ' & \side{' + nice.get(method, method.upper()) +'$^{' + nicerm(eval_name) + '}$} '
#     tabular += '\\\\\hline\n'
#
#     for dataset in datasets:
#         tabular += nice.get(dataset, dataset.upper()) + ' '
#         ranks_no_gap = []
#         for method in methods:
#             learner = 'lr' if not method.startswith('svm') else 'svmperf'
#             key = f'{dataset}-{method}-{learner}-{}-{eval_name}'
#             ranks_no_gap.append(stats[dataset][eval_name].get(key, (None, None, len(methods)))[2])
#         ranks_no_gap = sorted(ranks_no_gap)
#         ranks_no_gap = {rank:i+1 for i,rank in enumerate(ranks_no_gap)}
#         for method in methods:
#             learner = 'lr' if not method.startswith('svm') else 'svmperf'
#             key = f'{dataset}-{method}-{learner}-{sample_size}-{eval_name}'
#             if key in stats[dataset][eval_name]:
#                 _, _, abs_rank = stats[dataset][eval_name][key]
#                 real_rank = ranks_no_gap[abs_rank]
#                 tabular += f' & {real_rank}'
#                 tabular += color_from_abs_rank(real_rank, len(methods), maxtone=MAXTONE)
#             else:
#                 tabular += ' & --- '
#             old_rank = gao_seb_ranks.get(f'{dataset}-{method}-{eval_name}', 'error')
#             tabular += f' ({old_rank})'
#         tabular += '\\\\\hline\n'
#     tabular += "\end{tabularx}"
#     table += tabular + """
#         }
#       \end{center}
#       \label{tab:""" + eval_name + """ranks}
#     \end{table}
#     """
#     save_table(f'../tables/tab_rank_{eval_name}.tex', table)
#
#
# print("[Done]")