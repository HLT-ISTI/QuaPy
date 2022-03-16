import quapy as qp
from quapy.data import LabelledCollection
from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import join
import os
import pickle
from utils import *
from tqdm import tqdm
import shutil


vector_generation = 'average'

datapath = './data'
domain = f'Books-roberta-base-finetuned/checkpoint-1188-{vector_generation}'
outname = domain.replace('-finetuned', '-finetuned-pkl')

protocol = 'app'

print('pickling npy txt files')
print('from:', join(datapath, domain))
print('to', join(datapath, outname))
print('for protocol:', protocol)

os.makedirs(join(datapath, outname), exist_ok=True)
os.makedirs(join(datapath, outname, protocol), exist_ok=True)
os.makedirs(join(datapath, outname, protocol, 'dev_samples'), exist_ok=True)
os.makedirs(join(datapath, outname, protocol, 'test_samples'), exist_ok=True)
shutil.copyfile(join(datapath, domain, protocol, 'dev_prevalences.txt'), join(datapath, outname, protocol, 'dev_prevalences.txt'))
shutil.copyfile(join(datapath, domain, protocol, 'test_prevalences.txt'), join(datapath, outname, protocol, 'test_prevalences.txt'))


train = load_simple_sample_npytxt(join(datapath, domain), 'training_data', classes=np.arange(5))
pickle.dump(train, open(join(datapath, outname, 'training_data.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)


def transform_folder_samples(protocol, splitname):
    folder_dir=join(datapath, domain, protocol, splitname)
    for i, sample in tqdm(enumerate(load_samples_folder(folder_dir, filter=None, load_fn=load_simple_sample_npytxt, classes=train.classes_))):
        pickle.dump(sample, open(join(datapath, outname, protocol, splitname, f'{i}.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)


transform_folder_samples(protocol, 'dev_samples')
transform_folder_samples(protocol, 'test_samples')



