import os

import quapy as qp

os.makedirs('./latex_dataset', exist_ok=True)

REV_I = '\\review{'
REV_E = '}'
with open('./latex_dataset/tab_ucimulti.tex', 'wt') as foo:
    foo.write("""
    \\begin{tabular}{lrrrr}
    \\toprule
         \multicolumn{1}{l}{Dataset name} &  \multicolumn{1}{c}{\#training}&  \multicolumn{1}{c}{\#test} & \multicolumn{1}{c}{\#classes} & \multicolumn{1}{c}{\#features} \\\\
         \midrule
    
    """)

    for i, dataset in enumerate(qp.datasets.UCI_MULTICLASS_DATASETS):
        data = qp.datasets.fetch_UCIMulticlassDataset(dataset)
        dataset = dataset.replace('_', '\_')
        ntr = len(data.training)
        nte = len(data.test)
        n = data.n_classes
        nfeat = data.training.X.shape[1]
        if i < 5:
            foo.write(f'{dataset} & {ntr} & {nte} & {n} & {nfeat} \\\\')
        else:
            foo.write(f'{REV_I}{dataset}{REV_E} & {REV_I}{ntr}{REV_E} & {REV_I}{nte}{REV_E} & {REV_I}{n}{REV_E} & {REV_I}{nfeat}{REV_E} \\\\')
        foo.write('\n')

    foo.write("""
         \\bottomrule
         \end{tabular}
    """)



