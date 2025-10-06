# Datasets

QuaPy makes available several datasets that have been used in
quantification literature, as well as an interface to allow
anyone import their custom datasets.

A _Dataset_ object in QuaPy is roughly a pair of _LabelledCollection_ objects,
one playing the role of the training set, another the test set. 
_LabelledCollection_ is a data class consisting of the (iterable) 
instances and labels. This class handles most of the sampling functionality in QuaPy.
Take a look at the following code:

```python

import quapy as qp
import quapy.functional as F

instances = [
    '1st positive document', '2nd positive document',
    'the only negative document',
    '1st neutral document', '2nd neutral document', '3rd neutral document'
]
labels = [2, 2, 0, 1, 1, 1]

data = qp.data.LabelledCollection(instances, labels)
print(F.strprev(data.prevalence(), prec=2))

```

Output the class prevalences (showing 2 digit precision):
```
[0.17, 0.50, 0.33]
```

One can easily produce new samples at desired class prevalence values:

```python
sample_size = 10
prev = [0.4, 0.1, 0.5]
sample = data.sampling(sample_size, *prev)

print('instances:', sample.instances)
print('labels:', sample.labels)
print('prevalence:', F.strprev(sample.prevalence(), prec=2))
```

Which outputs: 
```
instances: ['the only negative document' '2nd positive document'
  '2nd positive document' '2nd neutral document' '1st positive document'
  'the only negative document' 'the only negative document'
  'the only negative document' '2nd positive document'
  '1st positive document']
labels: [0 2 2 1 2 0 0 0 2 2]
prevalence: [0.40, 0.10, 0.50]
```

Samples can be made consistent across different runs (e.g., to test
different methods on the same exact samples) by sampling and retaining
the indexes, that can then be used to generate the sample:

```python
index = data.sampling_index(sample_size, *prev)
for method in methods:
    sample = data.sampling_from_index(index)
    ...
```

However, generating samples for evaluation purposes is tackled in QuaPy
by means of the evaluation protocols (see the dedicated entries in the manuals
for [evaluation](./evaluation) and [protocols](./protocols)).


## Reviews Datasets

Three datasets of reviews about Kindle devices, Harry Potter's series, and
the well-known IMDb movie reviews can be fetched using a unified interface.
For example:

```python
import quapy as qp
data = qp.datasets.fetch_reviews('kindle')
```

These datasets have been used in:
```
Esuli, A., Moreo, A., & Sebastiani, F. (2018, October). 
A recurrent neural network for sentiment quantification. 
In Proceedings of the 27th ACM International Conference on 
Information and Knowledge Management (pp. 1775-1778).
```

The list of reviews ids is available in:

```python
qp.datasets.REVIEWS_SENTIMENT_DATASETS
```

Some statistics of the fhe available datasets are summarized below:

| Dataset | classes | train size | test size | train prev | test prev | type |
|---|:---:|:---:|:---:|:---:|:---:|---| 
| hp | 2 | 9533 | 18399 | \[0.018, 0.982\] | \[0.065, 0.935\] | text |
| kindle | 2 | 3821 | 21591 | \[0.081, 0.919\] | \[0.063, 0.937\] | text |
| imdb | 2 | 25000 | 25000 | \[0.500, 0.500\] | \[0.500, 0.500\] | text |

## Twitter Sentiment Datasets

11 Twitter datasets for sentiment analysis.
Text is not accessible, and the documents were made available
in tf-idf format. Each dataset presents two splits: a train/val
split for model selection purposes, and a train+val/test split
for model evaluation. The following code exemplifies how to load
a twitter dataset for model selection. 

```python
import quapy as qp
data = qp.datasets.fetch_twitter('gasp', for_model_selection=True)
```

The datasets were used in:

```
Gao, W., & Sebastiani, F. (2015, August). 
Tweet sentiment: From classification to quantification. 
In 2015 IEEE/ACM International Conference on Advances in 
Social Networks Analysis and Mining (ASONAM) (pp. 97-104). IEEE.
```

Three of the datasets (semeval13, semeval14, and semeval15) share the
same training set (semeval), meaning that the training split one would get
when requesting any of them is the same. The dataset "semeval" can only
be requested with "for_model_selection=True".
The lists of the Twitter dataset's ids can be consulted in:

```python
# a list of 11 dataset ids that can be used for model selection or model evaluation
qp.datasets.TWITTER_SENTIMENT_DATASETS_TEST

# 9 dataset ids in which "semeval13", "semeval14", and "semeval15" are replaced with "semeval"
qp.datasets.TWITTER_SENTIMENT_DATASETS_TRAIN  
```

Some details can be found below:

| Dataset | classes | train size | test size | features | train prev | test prev | type |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---| 
| gasp | 3 | 8788 | 3765 | 694582 | [0.421, 0.496, 0.082] | [0.407, 0.507, 0.086] | sparse |
| hcr | 3 | 1594 | 798 | 222046 | [0.546, 0.211, 0.243] | [0.640, 0.167, 0.193] | sparse |
| omd | 3 | 1839 | 787 | 199151 | [0.463, 0.271, 0.266] | [0.437, 0.283, 0.280] | sparse |
| sanders | 3 | 2155 | 923 | 229399 | [0.161, 0.691, 0.148] | [0.164, 0.688, 0.148] | sparse |
| semeval13 | 3 | 11338 | 3813 | 1215742 | [0.159, 0.470, 0.372] | [0.158, 0.430, 0.412] | sparse |
| semeval14 | 3 | 11338 | 1853 | 1215742 | [0.159, 0.470, 0.372] | [0.109, 0.361, 0.530] | sparse |
| semeval15 | 3 | 11338 | 2390 | 1215742 | [0.159, 0.470, 0.372] | [0.153, 0.413, 0.434] | sparse |
| semeval16 | 3 | 8000 | 2000 | 889504 | [0.157, 0.351, 0.492] | [0.163, 0.341, 0.497] | sparse |
| sst | 3 | 2971 | 1271 | 376132 | [0.261, 0.452, 0.288] | [0.207, 0.481, 0.312] | sparse |
| wa | 3 | 2184 | 936 | 248563 | [0.305, 0.414, 0.281] | [0.282, 0.446, 0.272] | sparse |
| wb | 3 | 4259 | 1823 | 404333 | [0.270, 0.392, 0.337] | [0.274, 0.392, 0.335] | sparse |


## UCI Machine Learning

### Binary datasets

A set of 32 datasets from the [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets.php) 
used in:

```
Pérez-Gállego, P., Quevedo, J. R., & del Coz, J. J. (2017).
Using ensembles for problems with characterizable changes 
in data distribution: A case study on quantification.
Information Fusion, 34, 87-100.
```

The list does not exactly coincide with that used in Pérez-Gállego et al. 2017
since we were unable to find the datasets with ids "diabetes" and "phoneme".

These dataset can be loaded by calling, e.g.:

```python
import quapy as qp

data = qp.datasets.fetch_UCIBinaryDataset('yeast', verbose=True)
```

This call will return a _Dataset_ object in which the training and
test splits are randomly drawn, in a stratified manner, from the whole
collection at 70% and 30%, respectively. The _verbose=True_ option indicates
that the dataset description should be printed in standard output.
The original data is not split, 
and some papers submit the entire collection to a kFCV validation. 
In order to accommodate with these practices, one could first instantiate
the entire collection, and then creating a generator that will return one
training+test dataset at a time, following a kFCV protocol:

```python
import quapy as qp

collection = qp.datasets.fetch_UCIBinaryLabelledCollection("yeast")
for data in qp.data.Dataset.kFCV(collection, nfolds=5, nrepeats=2):
  ...
```

Above code will allow to conduct a 2x5FCV evaluation on the "yeast" dataset.

All datasets come in numerical form (dense matrices); some statistics
are summarized below.

| Dataset | classes | instances | features | prev | type |
|---|:---:|:---:|:---:|:---:|---| 
| acute.a | 2 | 120 | 6 | [0.508, 0.492] | dense |
| acute.b | 2 | 120 | 6 | [0.583, 0.417] | dense |
| balance.1 | 2 | 625 | 4 | [0.539, 0.461] | dense |
| balance.2 | 2 | 625 | 4 | [0.922, 0.078] | dense |
| balance.3 | 2 | 625 | 4 | [0.539, 0.461] | dense |
| breast-cancer | 2 | 683 | 9 | [0.350, 0.650] | dense |
| cmc.1 | 2 | 1473 | 9 | [0.573, 0.427] | dense |
| cmc.2 | 2 | 1473 | 9 | [0.774, 0.226] | dense |
| cmc.3 | 2 | 1473 | 9 | [0.653, 0.347] | dense |
| ctg.1 | 2 | 2126 | 21 | [0.222, 0.778] | dense |
| ctg.2 | 2 | 2126 | 21 | [0.861, 0.139] | dense |
| ctg.3 | 2 | 2126 | 21 | [0.917, 0.083] | dense |
| german | 2 | 1000 | 24 | [0.300, 0.700] | dense |
| haberman | 2 | 306 | 3 | [0.735, 0.265] | dense |
| ionosphere | 2 | 351 | 34 | [0.641, 0.359] | dense |
| iris.1 | 2 | 150 | 4 | [0.667, 0.333] | dense |
| iris.2 | 2 | 150 | 4 | [0.667, 0.333] | dense |
| iris.3 | 2 | 150 | 4 | [0.667, 0.333] | dense |
| mammographic | 2 | 830 | 5 | [0.514, 0.486] | dense |
| pageblocks.5 | 2 | 5473 | 10 | [0.979, 0.021] | dense |
| semeion | 2 | 1593 | 256 | [0.901, 0.099] | dense |
| sonar | 2 | 208 | 60 | [0.534, 0.466] | dense |
| spambase | 2 | 4601 | 57 | [0.606, 0.394] | dense |
| spectf | 2 | 267 | 44 | [0.794, 0.206] | dense |
| tictactoe | 2 | 958 | 9 | [0.653, 0.347] | dense |
| transfusion | 2 | 748 | 4 | [0.762, 0.238] | dense |
| wdbc | 2 | 569 | 30 | [0.627, 0.373] | dense |
| wine.1 | 2 | 178 | 13 | [0.669, 0.331] | dense |
| wine.2 | 2 | 178 | 13 | [0.601, 0.399] | dense |
| wine.3 | 2 | 178 | 13 | [0.730, 0.270] | dense |
| wine-q-red | 2 | 1599 | 11 | [0.465, 0.535] | dense |
| wine-q-white | 2 | 4898 | 11 | [0.335, 0.665] | dense |
| yeast | 2 | 1484 | 8 | [0.711, 0.289] | dense |

#### Notes:
All datasets will be downloaded automatically the first time they are requested, and
stored in the _quapy_data_ folder for faster further reuse. 

However, notice that it is a good idea to ignore datasets:
* _acute.a_ and _acute.b_: these are very easy and many classifiers would score 100% accuracy
* _balance.2_: this is extremely difficult; probably there is some problem with this dataset, 
the errors it tends to produce are orders of magnitude greater than for other datasets, 
and this has a disproportionate impact in the average performance.

### Multiclass datasets

A collection of 24 multiclass datasets from the [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets.php). 
Some of the datasets were first used in [this paper](https://arxiv.org/abs/2401.00490) and can be instantiated as follows:

```python
import quapy as qp
data = qp.datasets.fetch_UCIMulticlassLabelledCollection('dry-bean', verbose=True)
```

A dataset can be instantiated filtering classes with a minimum number of instances using the `min_class_support` parameter
(default: `100`) as folows:


```python
import quapy as qp
data = qp.datasets.fetch_UCIMulticlassLabelledCollection('dry-bean', min_class_support=50, verbose=True)
```

There are no pre-defined train-test partitions for these datasets, but you can easily create your own with the
`split_stratified` method, e.g., `data.split_stratified()`. This can be also achieved using the method `fetch_UCIMulticlassDataset`
as shown below:

```python
data = qp.datasets.fetch_UCIMulticlassDataset('dry-bean', min_test_split=0.4, verbose=True)
train, test = data.train_test
```

This method tries to respect the `min_test_split` value while generating the train-test partition, but the resulting training set
will not be bigger than `max_train_instances`, which defaults to `25000`. A bigger value can be passed as a parameter:

```python
data = qp.datasets.fetch_UCIMulticlassDataset('dry-bean', min_test_split=0.4, max_train_instances=30000, verbose=True)
train, test = data.train_test
```

The datasets correspond to a part of the datasets that can be retrieved from the platform using the following filters: 
* datasets for classification
* more than 2 classes 
* containing at least 1,000 instances
* can be imported using the Python API. 

Some statistics about these datasets are displayed below :

| **Dataset** | **classes** | **instances** | **features** | **prevs** | **type** |
|:------------|:-----------:|:-------------:|:------------:|:----------|:--------:|
| dry-bean         |  7 |   13611 |  16 | [0.097, 0.038, 0.120, 0.261, 0.142, 0.149, 0.194] | dense |
| wine-quality     |  5 |    6462 |  11 | [0.033, 0.331, 0.439, 0.167, 0.030] | dense |
| academic-success |  3 |    4424 |  36 | [0.321, 0.179, 0.499] | dense |
| digits           | 10 |    5620 |  64 | [0.099, 0.102, 0.099, 0.102, 0.101, 0.099, 0.099, 0.101, 0.099, 0.100] | dense |
| letter           | 26 |   20000 |  16 | [0.039, 0.038, 0.037, 0.040, 0.038, 0.039, 0.039, 0.037, 0.038, 0.037, 0.037, 0.038, 0.040, 0.039, 0.038, 0.040, 0.039, 0.038, 0.037, 0.040, 0.041, 0.038, 0.038, 0.039, 0.039, 0.037] | dense |
| abalone          | 11 |    3842 |   9 | [0.030, 0.067, 0.102, 0.148, 0.179, 0.165, 0.127, 0.069, 0.053, 0.033, 0.027] | dense |
| obesity          |  7 |    2111 |  23 | [0.129, 0.136, 0.166, 0.141, 0.153, 0.137, 0.137] | dense |
| nursery          |  4 |   12958 |  19 | [0.333, 0.329, 0.312, 0.025] | dense |
| yeast            |  4 |    1299 |   8 | [0.356, 0.125, 0.188, 0.330] | dense |
| hand_digits      | 10 |   10992 |  16 | [0.104, 0.104, 0.104, 0.096, 0.104, 0.096, 0.096, 0.104, 0.096, 0.096] | dense |
| satellite        |  6 |    6435 |  36 | [0.238, 0.109, 0.211, 0.097, 0.110, 0.234] | dense |
| shuttle          |  4 |   57927 |   7 | [0.787, 0.003, 0.154, 0.056] | dense |
| cmc              |  3 |    1473 |   9 | [0.427, 0.226, 0.347] | dense |
| isolet           | 26 |    7797 | 617 | [0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038, 0.038] | dense |
| waveform-v1      |  3 |    5000 |  21 | [0.331, 0.329, 0.339] | dense |
| molecular        |  3 |    3190 | 227 | [0.240, 0.241, 0.519] | dense |
| poker_hand       |  8 | 1024985 |  10 | [0.501, 0.423, 0.048, 0.021, 0.004, 0.002, 0.001, 0.000] | dense |
| connect-4        |  3 |   67557 |  84 | [0.095, 0.246, 0.658] | dense |
| mhr              |  3 |    1014 |   6 | [0.268, 0.400, 0.331] | dense |
| chess            | 15 |   27870 |  20 | [0.100, 0.051, 0.102, 0.078, 0.017, 0.007, 0.163, 0.061, 0.025, 0.021, 0.014, 0.071, 0.150, 0.129, 0.009] | dense |
| page_block       |  3 |    5357 |  10 | [0.917, 0.061, 0.021] | dense |
| phishing         |  3 |    1353 |   9 | [0.519, 0.076, 0.405] | dense |
| image_seg        |  7 |    2310 |  19 | [0.143, 0.143, 0.143, 0.143, 0.143, 0.143, 0.143] | dense |
| hcv              |  4 |    1385 |  28 | [0.243, 0.240, 0.256, 0.261] | dense |

Values shown above refer to datasets obtained through `fetchUCIMulticlassLabelledCollection` using all default parameters.

## LeQua 2022 Datasets

QuaPy also provides the datasets used for the LeQua 2022 competition.
In brief, there are 4 tasks (T1A, T1B, T2A, T2B) having to do with text quantification
problems. Tasks T1A and T1B provide documents in vector form, while T2A and T2B provide 
raw documents instead.
Tasks T1A and T2A are binary sentiment quantification problems, while T2A and T2B 
are multiclass quantification problems consisting of estimating the class prevalence 
values of 28 different merchandise products.

Every task consists of a training set, a set of validation samples (for model selection)
and a set of test samples (for evaluation). QuaPy returns this data as a LabelledCollection
(training) and two generation protocols (for validation and test samples), as follows:

```python
training, val_generator, test_generator = fetch_lequa2022(task=task)
```

See the `lequa2022_experiments.py` in the examples folder for further details on how to
carry out experiments using these datasets.  

The datasets are downloaded only once, and stored for fast reuse.

Some statistics are summarized below:

| Dataset | classes | train size | validation samples | test samples |  docs by sample  |   type   |
|---------|:-------:|:----------:|:------------------:|:------------:|:----------------:|:--------:| 
| T1A     |    2    |    5000    |        1000        |     5000     |       250        |  vector  | 
| T1B     |   28    |   20000    |        1000        |     5000     |       1000       |  vector  |
| T2A     |    2    |    5000    |        1000        |     5000     |       250        |   text   |
| T2B     |   28    |   20000    |        1000        |     5000     |       1000       |   text   |

For further details on the datasets, we refer to the original 
[paper](https://ceur-ws.org/Vol-3180/paper-146.pdf):

```
Esuli, A., Moreo, A., Sebastiani, F., & Sperduti, G. (2022).
A Detailed Overview of LeQua@ CLEF 2022: Learning to Quantify.
```

## IFCB Plankton dataset

IFCB is a dataset of plankton species in water samples hosted in `Zenodo <https://zenodo.org/records/10036244>`_.
This dataset is based on the data available publicly at `WHOI-Plankton repo <https://github.com/hsosik/WHOI-Plankton>`_
and in the scripts for the processing are available at `P. González's repo <https://github.com/pglez82/IFCB_Zenodo>`_.

This dataset comes with precomputed features for testing quantification algorithms.

Some statistics:

|                 | **Training** | **Validation** | **Test** |
|-----------------|:------------:|:--------------:|:--------:|
| samples         |      200     |       86       |    678   |
| total instances |    584474    |     246916     |  2626429 |
| mean per sample |    2922.3    |     2871.1     |  3873.8  |
| min per sample  |      266     |       59       |    33    |
| max per sample  |     6645     |      7375      |   9112   |

The number of features is 512, while the number of classes is 50.
In terms of prevalence, the mean is 0.020, the minimum is 0, and the maximum is 0.978.

The dataset can be loaded for model selection (`for_model_selection=True`, thus returning the training and validation) 
or for test (`for_model_selection=False`, thus returning the training+validation and the test). 

Additionally, the training can be interpreted as a list (a generator) of samples (`single_sample_train=False`)
or as a single training set (`single_sample_train=True`).

Example:

```python
train, val_gen = qp.datasets.fetch_IFCB(for_model_selection=True, single_sample_train=True)
# ... model selection

train, test_gen = qp.datasets.fetch_IFCB(for_model_selection=False, single_sample_train=True)
# ... train and evaluation
```



## Adding Custom Datasets

QuaPy provides data loaders for simple formats dealing with 
text, following the format:

```
class-id \t first document's pre-processed text \n
class-id \t second document's pre-processed text \n
...
```

and sparse representations of the form:

```
{-1, 0, or +1} col(int):val(float) col(int):val(float) ... \n
...
```

The code in charge in loading a LabelledCollection is:

```python
@classmethod
def load(cls, path:str, loader_func:callable):
    return LabelledCollection(*loader_func(path))
```

indicating that any _loader_func_ (e.g., a user-defined one) which 
returns valid arguments for initializing a _LabelledCollection_ object will allow
to load any collection. In particular, the _LabelledCollection_ receives as 
arguments the instances (as an iterable) and the labels (as an iterable) and,
additionally, the number of classes can be specified (it would otherwise be
inferred from the labels, but that requires at least one positive example for
all classes to be present in the collection).

The same _loader_func_ can be passed to a Dataset, along with two 
paths, in order to create a training and test pair of _LabelledCollection_,
e.g.:

```python
import quapy as qp

train_path = '../my_data/train.dat'
test_path = '../my_data/test.dat'

def my_custom_loader(path):
    with open(path, 'rb') as fin:
        ...
    return instances, labels

data = qp.data.Dataset.load(train_path, test_path, my_custom_loader)
```

### Data Processing

QuaPy implements a number of preprocessing functions in the package _qp.data.preprocessing_, including:

* _text2tfidf_: tfidf vectorization 
* _reduce_columns_: reducing the number of columns based on term frequency
* _standardize_: transforms the column values into z-scores (i.e., subtract the mean and normalizes by the standard deviation, so
that the column values have zero mean and unit variance).
* _index_: transforms textual tokens into lists of numeric ids) 
