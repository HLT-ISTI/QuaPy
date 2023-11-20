import pandas as pd

report = pd.read_csv('../results_lequa_mrae/KDEy-MLE.dataframe')

means = report.mean()
print(f'KDEy-MLE\tLeQua-T1B\t{means["mae"]:.5f}\t{means["mrae"]:.5f}\t{means["kld"]:.5f}\n')