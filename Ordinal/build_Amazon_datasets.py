import gzip
import quapy as qp
from quapy.data import LabelledCollection
import quapy.functional as F
import os
from os.path import join
from pathlib import Path


datadir = '/mnt/1T/Datasets/Amazon/reviews'
outdir  = './data/'
domain = 'Books'
seed = 7

tr_size = 20000
val_size = 1000
te_size = 1000
nval = 1000
nte = 5000

# domain = 'Gift_Cards'
# tr_size = 200
# val_size = 100
# te_size = 100
# nval = 20
# nte = 40


def from_gz_text(path, encoding='utf-8', class2int=True):
    """
    Reads a labelled colletion of documents.
    File fomart <0-4>\t<document>\n

    :param path: path to the labelled collection
    :param encoding: the text encoding used to open the file
    :return: a list of sentences, and a list of labels
    """
    all_sentences, all_labels = [], []
    file = gzip.open(path, 'rt', encoding=encoding).readlines()
    for line in file:
        line = line.strip()
        if line:
            try:
                label, sentence = line.split('\t')
                sentence = sentence.strip()
                if class2int:
                    label = int(label) - 1
                if label >= 0:
                    if sentence:
                        all_sentences.append(sentence)
                        all_labels.append(label)
            except ValueError:
                print(f'format error in {line}')
    return all_sentences, all_labels


def write_txt_sample(sample: LabelledCollection, path):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, 'wt') as foo:
        for document, label in zip(*sample.Xy):
            foo.write(f'{label}\t{document}\n')


def gen_samples_APP(pool: LabelledCollection, nsamples, sample_size, outdir, prevpath):
    os.makedirs(outdir, exist_ok=True)
    with open(prevpath, 'wt') as prevfile:
        prevfile.write('id,' + ','.join(f'{c}' for c in pool.classes_) + '\n')
        for i, prev in enumerate(F.uniform_simplex_sampling(n_classes=pool.n_classes, size=nsamples)):
            sample = pool.sampling(sample_size, *prev)
            write_txt_sample(sample, join(outdir, f'{i}.txt'))
            prevfile.write(f'{i},' + ','.join(f'{p:.3f}' for p in sample.prevalence()) + '\n')


def gen_samples_NPP(pool: LabelledCollection, nsamples, sample_size, outdir, prevpath):
    os.makedirs(outdir, exist_ok=True)
    with open(prevpath, 'wt') as prevfile:
        prevfile.write('id,' + ','.join(f'{c}' for c in pool.classes_) + '\n')
        for i, sample in enumerate(pool.natural_sampling_generator(sample_size, repeats=nsamples)):
            write_txt_sample(sample, join(outdir, f'{i}.txt'))
            prevfile.write(f'{i},' + ','.join(f'{p:.3f}' for p in sample.prevalence()) + '\n')



fullpath = join(datadir,domain)+'.txt.gz'

data = LabelledCollection.load(fullpath, from_gz_text)
print(len(data))
print(data.classes_)
print(data.prevalence())

with qp.util.temp_seed(seed):
    train, rest = data.split_stratified(train_prop=tr_size)

    devel, test = rest.split_stratified(train_prop=0.5)
    print(len(train))
    print(len(devel))
    print(len(test))

    domaindir = join(outdir, domain)

    write_txt_sample(train, join(domaindir, 'training_data.txt'))
    write_txt_sample(devel, join(domaindir, 'development_data.txt'))
    write_txt_sample(test, join(domaindir, 'test_data.txt'))

    gen_samples_APP(devel, nsamples=nval, sample_size=val_size, outdir=join(domaindir, 'app', 'dev_samples'),
                    prevpath=join(domaindir, 'app', 'dev_prevalences.txt'))
    gen_samples_APP(test, nsamples=nte, sample_size=te_size, outdir=join(domaindir, 'app', 'test_samples'),
                    prevpath=join(domaindir, 'app', 'test_prevalences.txt'))

    gen_samples_NPP(devel, nsamples=nval, sample_size=val_size, outdir=join(domaindir, 'npp', 'dev_samples'),
                    prevpath=join(domaindir, 'npp', 'dev_prevalences.txt'))
    gen_samples_NPP(test, nsamples=nte, sample_size=te_size, outdir=join(domaindir, 'npp', 'test_samples'),
                    prevpath=join(domaindir, 'npp', 'test_prevalences.txt'))



