import pickle
import os

dataset = 'lequa/T1B'
for metric in ['mae', 'mrae']:
    print('metric', metric)
    for method in ['KDEy-DMhd4', 'KDEy-DMhd4+', 'KDEy-DMhd4++']:

        path = f'/home/moreo/QuaPy/distribution_matching/results/{dataset}/{metric}/{method}.hyper.pkl'
        if os.path.exists(path):
            obj = pickle.load(open(path, 'rb'))
            print(method, obj)
        else:
            print(f'path {path} does not exist')

    print()



