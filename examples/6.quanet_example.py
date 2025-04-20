import quapy as qp
from quapy.classification.neural import CNNnet
from quapy.classification.neural import NeuralClassifierTrainer
from quapy.method.meta import QuaNet
import quapy.functional as F

"""
This example shows how to train QuaNet. The internal classifier is a word-based CNN. 
"""

# set the sample size in the environment
qp.environ["SAMPLE_SIZE"] = 100

# the dataset is textual (Kindle reviews from Amazon), so we need to index terms, i.e.,
# we need to convert distinct terms into numerical ids
dataset = qp.datasets.fetch_reviews('kindle', pickle=True)
qp.data.preprocessing.index(dataset, min_df=5, inplace=True)
train, test = dataset.train_test

# train the text classifier:
cnn_module = CNNnet(dataset.vocabulary_size, dataset.training.n_classes)
cnn_classifier = NeuralClassifierTrainer(cnn_module, device='cuda')
cnn_classifier.fit(*dataset.training.Xy)

# train QuaNet (alternatively, we can set fit_classifier=True and let QuaNet train the classifier)
quantifier = QuaNet(cnn_classifier, device='cuda')
quantifier.fit(train, fit_classifier=False)

# prediction and evaluation
estim_prevalence = quantifier.predict(test.instances)
mae = qp.error.mae(test.prevalence(), estim_prevalence)

print(f'true prevalence: {F.strprev(test.prevalence())}')
print(f'estim prevalence: {F.strprev(estim_prevalence)}')
print(f'MAE = {mae:.4f}')