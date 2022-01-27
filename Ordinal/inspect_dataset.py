import quapy as qp
from quapy.data import LabelledCollection
from quapy.data.reader import from_text
from quapy.functional import strprev

category = 'Books'
datadir = './data'

training_path = f'{datadir}/{category}/training_data.txt'

data = LabelledCollection.load(training_path, loader_func=from_text)

print(len(data))
print(strprev(data.prevalence()))


