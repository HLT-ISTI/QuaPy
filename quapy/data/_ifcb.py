import os
import pandas as pd
import math
from typing import Optional
from quapy.data import LabelledCollection
from quapy.protocol import AbstractProtocol
from pathlib import Path


def get_sample_list(path_dir):
    """
    Gets a sample list finding the csv files in a directory

    :param path_dir: directory to look for samples
    :return: list of samples
    """
    samples = []
    for filename in sorted(os.listdir(path_dir)):
        if filename.endswith('.csv'):
            samples.append(filename)
    return samples


def generate_modelselection_split(samples, test_prop=0.3):
    """This function generates a train/test partition for model selection
    without the use of random numbers so the split is always the same

    :param samples: list of samples
    :param test_prop: float, percentage saved for test. Defaults to 0.3.
    :return: list of samples to use as train and list of samples to use as test
    """
    num_items_to_pick = math.ceil(len(samples) * test_prop)
    step_size = math.floor(len(samples) / num_items_to_pick)
    test_indices = [i * step_size for i in range(num_items_to_pick)]
    test = [samples[i] for i in test_indices]
    train = [item for i, item in enumerate(samples) if i not in test_indices]
    return train, test


class IFCBTrainSamplesFromDir(AbstractProtocol):

    def __init__(self, path_dir:str, classes: list, samples: list = None):
        self.path_dir = path_dir
        self.classes = classes
        self.samples = []
        if samples is not None:
            self.samples = samples
        else:
            self.samples = get_sample_list(path_dir)

    def __call__(self):
        for sample in self.samples:
            s = pd.read_csv(os.path.join(self.path_dir,sample))
            # all columns but the first where we get the class
            X = s.iloc[:, 1:].to_numpy()
            y = s.iloc[:, 0].to_numpy()
            yield LabelledCollection(X, y, classes=self.classes)

    def total(self):
        """
        Returns the total number of samples that the protocol generates.

        :return: The number of training samples to generate.
        """
        return len(self.samples)


class IFCBTestSamples(AbstractProtocol):

    def __init__(self, path_dir:str, test_prevalences: Optional[pd.DataFrame]=None, samples: list=None, classes: list=None):
        self.path_dir = path_dir
        self.test_prevalences = test_prevalences
        self.classes = classes
        if samples is not None:
            self.samples = samples
        else:
            self.samples = get_sample_list(path_dir)

    def __call__(self):
        for test_sample in self.samples:
            s = pd.read_csv(os.path.join(self.path_dir,test_sample))
            if self.test_prevalences is not None:
                X = s
                # If we are working with the test samples, we have a dataframe with the prevalences and no labels for the test
                prevalences = self.test_prevalences.loc[self.test_prevalences['sample']==Path(test_sample).stem].to_numpy()[:,1:].flatten().astype(float)
            else:
                X = s.iloc[:, 1:].to_numpy()
                y = s.iloc[:,0]
                # In this case we compute the sample prevalences from the labels
                prevalences = y[y.isin(self.classes)].value_counts().reindex(self.classes, fill_value=0).to_numpy()/len(s)
            yield X, prevalences

    def total(self):
        """
        Returns the total number of samples that the protocol generates.

        :return: The number of training samples to generate.
        """
        return len(self.samples)
