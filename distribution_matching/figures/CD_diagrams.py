import pandas as pd
from critdd import Diagram
from distribution_matching.commons import BIN_METHODS
import quapy as qp

error = 'mrae'

datasets = qp.datasets.UCI_BINARY_DATASETS
datasets_remove = ['iris.1', 'acute.a', 'acute.b'] 
for d in datasets_remove:
    if d in datasets:
        datasets.remove(d)

methods = BIN_METHODS
methods.remove("MS")
methods.remove("EMQ-BCTS")

results = pd.DataFrame(columns=methods,index=datasets,dtype=float)


for method in methods:
    for dataset in datasets:
        df=pd.read_csv('binary/'+error+'/'+method+'_'+dataset+'.dataframe',sep=',')
        results[method][dataset]=float(df[error].mean())

print(results.columns)

diagram = Diagram(
    results.to_numpy(),
    treatment_names = results.columns,
    maximize_outcome = False
)

file_name = "CD_bin_"+error+".tex"

diagram.to_file(
    file_name,
    alpha = .05,
    adjustment = "holm",
    reverse_x = True,
    axis_options = {"title": error.upper()},
)

# Open the file in read mode
with open(file_name, 'r') as file:
    # Read the file content
    content = file.read()

modified_content = content.replace('EDy+', 'EDy')
modified_content = modified_content.replace('PACC+', 'PACCy')
modified_content = modified_content.replace('ACC+', 'ACC')


# Open the file in write mode and write the modified content
with open(file_name, 'w') as file:
    file.write(modified_content)