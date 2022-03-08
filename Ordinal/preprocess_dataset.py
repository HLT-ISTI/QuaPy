import quapy as qp
from quapy.data import LabelledCollection
from sklearn.feature_extraction.text import TfidfVectorizer
from os.path import join
import os
import pickle
from utils import load_samples
from tqdm import tqdm
import shutil


datapath = './data'
domain = 'Books'
outname = domain + '-tfidf'

def save_preprocessing_info(transformer):
    with open(join(datapath, outname, 'prep-info.txt'), 'wt') as foo:
        foo.write(f'{str(transformer)}\n')


os.makedirs(join(datapath, outname), exist_ok=True)
os.makedirs(join(datapath, outname, 'app'), exist_ok=True)
os.makedirs(join(datapath, outname, 'app', 'dev_samples'), exist_ok=True)
os.makedirs(join(datapath, outname, 'app', 'test_samples'), exist_ok=True)
shutil.copyfile(join(datapath, domain, 'app', 'dev_prevalences.txt'), join(datapath, outname, 'app', 'dev_prevalences.txt'))
shutil.copyfile(join(datapath, domain, 'app', 'test_prevalences.txt'), join(datapath, outname, 'app', 'test_prevalences.txt'))
os.makedirs(join(datapath, outname, 'npp'), exist_ok=True)
os.makedirs(join(datapath, outname, 'npp', 'dev_samples'), exist_ok=True)
os.makedirs(join(datapath, outname, 'npp', 'test_samples'), exist_ok=True)
shutil.copyfile(join(datapath, domain, 'npp', 'dev_prevalences.txt'), join(datapath, outname, 'npp', 'dev_prevalences.txt'))
shutil.copyfile(join(datapath, domain, 'npp', 'test_prevalences.txt'), join(datapath, outname, 'npp', 'test_prevalences.txt'))


tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), min_df=5)

train = LabelledCollection.load(join(datapath, domain, 'training_data.txt'), loader_func=qp.data.reader.from_text)
train.instances = tfidf.fit_transform(train.instances)
save_preprocessing_info(tfidf)
pickle.dump(train, open(join(datapath, outname, 'training_data.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)


def transform_folder_samples(protocol, splitname):
    for i, sample in tqdm(enumerate(load_samples(join(datapath, domain, protocol, splitname), classes=train.classes_))):
        sample.instances = tfidf.transform(sample.instances)
        pickle.dump(sample, open(join(datapath, outname, protocol, splitname, f'{i}.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)


transform_folder_samples('app', 'dev_samples')
transform_folder_samples('app', 'test_samples')
transform_folder_samples('npp', 'dev_samples')
transform_folder_samples('npp', 'test_samples')



