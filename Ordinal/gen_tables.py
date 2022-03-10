import pandas as pd
from os.path import join
import os
from glob import glob
from pathlib import Path

from Ordinal.main import quantifiers
from Ordinal.tabular import Table

domain = 'Books-tfidf'
prot = 'app'
outpath = f'./tables/{domain}/{prot}/results.tex'

resultpath = join('./results', domain, prot)

methods = [qname for qname, *_ in quantifiers()]
# methods += [m+'-r' for m in methods]

table = Table(benchmarks=['low', 'mid', 'high', 'all'], methods=methods, prec_mean=4, show_std=True, prec_std=4)


for resultfile in glob(f'{resultpath}/*.csv'):
    df = pd.read_csv(resultfile)
    nmd = df['nmd'].values
    resultname = Path(resultfile).name
    method, drift, *other = resultname.replace('.csv', '').split('.')
    if other:
        continue
        method += '-r'

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

