import numpy as np


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
    'epaccmaeptr': 'E(PACC)$_\mathrm{Ptr}$',
    'epaccmaemae': 'E(PACC)$_\mathrm{AE}$',
    'epaccmraeptr': 'E(PACC)$_\mathrm{Ptr}$',
    'epaccmraemrae': 'E(PACC)$_\mathrm{RAE}$',
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


def nicename(method, eval_name=None, side=False):
    m = nice.get(method, method.upper())
    if eval_name is not None:
        o = '$^{' + nicerm(eval_name) + '}$'
        m = (m+o).replace('$$','')
    if side:
        m = '\side{'+m+'}'
    return m


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
