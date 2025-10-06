import quapy as qp
from quapy.method.aggregative import PACC
from quapy.data import LabelledCollection, Dataset
from quapy.protocol import ArtificialPrevalenceProtocol
import quapy.functional as F
import os
from os.path import join

# While quapy comes with ready-to-use datasets for experimental purposes, you may prefer to run experiments using
# your own data. Most of the quapy's functionality relies on an internal class called LabelledCollection, for fast
# indexing and sampling, and so this example provides guidance on how to convert your datasets into a LabelledCollection
# so all the functionality becomes available. This includes procedures for tuning the hyperparameters of your methods,
# evaluating the performance using high level sampling protocols, etc.

# Let us assume that we have a binary sentiment dataset of opinions in natural language. We will use the "IMDb"
# dataset of reviews, which can be downloaded as follows
URL_TRAIN = f'https://zenodo.org/record/4117827/files/imdb_train.txt'
URL_TEST = f'https://zenodo.org/record/4117827/files/imdb_test.txt'
os.makedirs('./reviews', exist_ok=True)
train_path = join('reviews', 'hp_train.txt')
test_path = join('reviews', 'hp_test.txt')
qp.util.download_file_if_not_exists(URL_TRAIN, train_path)
qp.util.download_file_if_not_exists(URL_TEST, test_path)

# these files contain 2 columns separated by a \t:
# the first one is a binary value (0=negative, 1=positive), and the second is the text
# Everything we need is to implement a function returning the instances and the labels as follows
def my_data_loader(path):
    with open(path, 'rt') as fin:
        labels, texts = zip(*[line.split('\t') for line in fin.readlines()])
        labels = list(map(int, labels))  # convert string numbers to int
        return texts, labels

# check that our function is working properly...
train_texts, train_labels = my_data_loader(train_path)
for i, (text, label) in enumerate(zip(train_texts, train_labels)):
    print(f'#{i}: {label=}\t{text=}')
    if i>=5:
        print('...')
        break

# We can now instantiate a LabelledCollection simply as
train_lc = LabelledCollection(instances=train_texts, labels=train_labels)
print('my training collection:', train_lc)

# We can instantiate directly a LabelledCollection using the data loader function,
# without having to load the data ourselves:
train_lc = LabelledCollection.load(train_path, loader_func=my_data_loader)
print('my training collection:', train_lc)

# We can do the same for the test set, or we can instead directly instantiate a Dataset object (this is by and large
# simply a tuple with training and test LabelledCollections) as follows:
my_data = Dataset.load(train_path, test_path, loader_func=my_data_loader)
print('my dataset:', my_data)

# However, since this is a textual dataset, we must vectorize it prior to training any quantification algorithm.
# We can do this in several ways in quapy. For example, manually...
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(min_df=5)
# Xtr = tfidf.fit_transform(my_data.training.instances)
# Xte = tfidf.transform(my_data.test.instances)
# ... or using some preprocessing functionality of quapy (recommended):
my_data_tfidf = qp.data.preprocessing.text2tfidf(my_data, min_df=5)

training, test = my_data_tfidf.train_test

# Once you have loaded your training and test data, you have access to a series of quapy's utilities, e.g.:
print(f'the training prevalence is {F.strprev(training.prevalence())}')
print(f'the test prevalence is {F.strprev(test.prevalence())}')
print(f'let us generate a small balanced training sample:')
desired_size = 200
desired_prevalence = [0.5, 0.5]
small_training_balanced = training.sampling(desired_size, *desired_prevalence, shuffle=True, random_state=0)
print(small_training_balanced)
print(f'or generating train/val splits such as: {training.split_stratified(train_prop=0.7)}')

# training
print('let us train a simple quantifier:...')
Xtr, ytr = training.Xy
quantifier = PACC()
quantifier.fit(Xtr, ytr)  # or: quantifier.fit(*training.Xy)

# test
print("and use quapy' evaluation functions")
evaluation_protocol = ArtificialPrevalenceProtocol(
    data=test,
    sample_size=200,
    random_state=0
)

report = qp.evaluation.evaluation_report(quantifier, protocol=evaluation_protocol, error_metrics=['ae'])
print(report)
print(f'mean absolute error across {len(report)} experiments: {report.mean(numeric_only=True)}')










