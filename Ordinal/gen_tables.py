import pandas as pd
from os.path import join
import os
from glob import glob
from pathlib import Path

from Ordinal.main import quantifiers
from Ordinal.tabular import Table

domain = 'Books-tfidf'
domain_bert_last = 'Books-roberta-base-finetuned-pkl/checkpoint-1188-last'
domain_bert_ave  = 'Books-roberta-base-finetuned-pkl/checkpoint-1188-average'
prot = 'app'
outpath = f'./tables/{domain}/{prot}/results.tex'

resultpath = join('./results', domain, prot)
resultpath_bertlast = join('./results', domain_bert_last, prot)
resultpath_bertave = join('./results', domain_bert_ave, prot)

methods = [qname for qname, *_ in quantifiers()]
methods_Rlast = [m+'-RoBERTa-last' for m in methods]
methods_Rave = [m+'-RoBERTa-average' for m in methods]
methods = methods + methods_Rlast + methods_Rave 
methods += [m+'-r' for m in methods]

table = Table(benchmarks=['low', 'mid', 'high', 'all'], methods=methods, prec_mean=4, show_std=True, prec_std=4)

resultfiles = list(glob(f'{resultpath}/*.csv')) + list(glob(f'{resultpath_bertlast}/*.csv')) + list(glob(f'{resultpath_bertave}/*.csv'))

for resultfile in resultfiles:
    df = pd.read_csv(resultfile)
    nmd = df['nmd'].values
    resultname = Path(resultfile).name
    method, drift, *other = resultname.replace('.csv', '').split('.')
    if other:
        method += '-r'
    if method not in methods:
        continue  

    table.add(drift, method, nmd)

os.makedirs(Path(outpath).parent, exist_ok=True)

tabular = """
    \\resizebox{\\textwidth}{!}{%
            \\begin{tabular}{|c||""" + ('c|' * (table.nbenchmarks)) + """} \hline
            """
tabular += table.latexTabularT(average=False)
tabular += """
    \end{tabular}%
    }"""

with open(outpath, 'wt') as foo:
    foo.write(tabular)
    foo.write('\n')

print('[done]')

