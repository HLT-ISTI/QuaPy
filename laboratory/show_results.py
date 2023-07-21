import sys
from pathlib import Path
import pandas as pd

#result_dir = 'results_tweet_1000'
result_dir = 'results_lequa'

dfs = []

pathlist = Path(result_dir).rglob('*.csv')
for path in pathlist:
     path_in_str = str(path)
     print(path_in_str)

     try:
          df = pd.read_csv(path_in_str, sep='\t')
          if not df.empty:
               dfs.append(df)
     except Exception:
          print('empty')

df = pd.concat(dfs)

for err in ['MAE', 'MRAE']:
     print('-'*100)
     print(err)
     print('-'*100)
     piv = df.pivot_table(index='Dataset', columns='Method', values=err)
     piv.loc['mean'] = piv.mean()

     pd.set_option('display.max_columns', None)
     pd.set_option('display.max_rows', None)
     pd.set_option('expand_frame_repr', False)
     print(piv)
     print()






