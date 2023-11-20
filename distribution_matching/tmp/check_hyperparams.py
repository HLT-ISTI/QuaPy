import pickle

dataset = 'lequa/T1A'
for metric in ['mae', 'mrae']:
    method1 = 'KDEy-closed++'
    # method2 = 'KDEy-DMhd3+'
    # method1 = 'KDEy-ML'
    # method2 = 'KDEy-ML+'

    path = f'../results/{dataset}/{metric}/{method1}.hyper.pkl'
    obj = pickle.load(open(path, 'rb'))
    print(method1, obj)

    # path = f'../results/{dataset}/{metric}/{method2}.hyper.pkl'
    # obj = pickle.load(open(path, 'rb'))
    # print(method2, obj)

