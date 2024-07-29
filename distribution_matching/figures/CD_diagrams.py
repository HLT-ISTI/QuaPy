import pandas as pd
from critdd import Diagram
from distribution_matching.commons import BIN_METHODS, METHODS
import quapy as qp
import importlib
from Orange.evaluation import scoring
_ = importlib.reload( scoring )



def generate_cd(dataset_global, error):
    if dataset_global=='binary':
        datasets = qp.datasets.UCI_BINARY_DATASETS.copy()
        datasets_remove = ['iris.1', 'acute.a', 'acute.b'] 
        for d in datasets_remove:
            if d in datasets:
                datasets.remove(d)
        methods = BIN_METHODS.copy()
        methods.remove("EMQ-BCTS")
        print(methods)
    elif dataset_global=='ucimulti':
        datasets = qp.datasets.UCI_MULTICLASS_DATASETS.copy()
        methods = METHODS.copy()
        methods.remove("EMQ-BCTS")
    elif dataset_global=='tweet':
        datasets = qp.datasets.TWITTER_SENTIMENT_DATASETS_TEST.copy()
        methods = METHODS.copy()
        methods.remove("EMQ-BCTS")
    elif dataset_global=='multimix':
        datasets = qp.datasets.UCI_MULTICLASS_DATASETS+qp.datasets.TWITTER_SENTIMENT_DATASETS_TEST+['lequa']
        methods = METHODS.copy()
        methods.remove("EMQ-BCTS")



    results = pd.DataFrame(columns=methods,index=datasets,dtype=float)


    for method in methods:
        for dataset in datasets:
            df=pd.read_csv('results/'+dataset_global+'/'+error+'/'+method+'_'+dataset+'.dataframe',sep=',')
            results[method][dataset]=float(df[error].mean())

    results = results.rename(columns={'ACC+': 'ACC', 'PACC+':'PACC','EDy+':'EDy'})

    diagram = Diagram(
        results.to_numpy(),
        treatment_names = results.columns,
        maximize_outcome = False
    )

    file_name = "CD_"+dataset_global+"_"+error+".tex"

    diagram.to_file(
        file_name,
        alpha = .05,
        adjustment = "holm",
        reverse_x = True,
        axis_options = {"title": error.upper()},
    )

    print(results)

    ranked_results = results.rank(axis=1, method='min', ascending=True)
    # Calculate the average rank for each method
    average_ranks = ranked_results.mean(axis=0)

    names = average_ranks.index
    avranks = average_ranks.to_numpy()
    cd = scoring.compute_CD(avranks, len(ranked_results))
    file_name = "CDOrange_"+dataset_global+"_"+error+".png"
    scoring.graph_ranks(avranks, names, cd, width=6, textspace=1.5,filename=file_name)

generate_cd('ucimulti','mae')
generate_cd('ucimulti','mrae')
generate_cd('binary','mae')
generate_cd('binary','mrae')
generate_cd('tweet','mae')
generate_cd('tweet','mrae')
generate_cd('multimix','mae')
generate_cd('multimix','mrae')