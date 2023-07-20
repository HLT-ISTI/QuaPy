import sys
from pathlib import Path
import pandas as pd

result_dir = 'results'

dfs = []

pathlist = Path(result_dir).rglob('*.csv')
for path in pathlist:
     path_in_str = str(path)
     print(path_in_str)

     df = pd.read_csv(path_in_str, sep='\t')

     dfs.append(df)

df = pd.concat(dfs)

piv = df.pivot_table(index='Dataset', columns='Method', values='MRAE')
piv.loc['mean'] = piv.mean()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)
print(piv)






