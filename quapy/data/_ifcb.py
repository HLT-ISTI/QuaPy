import os
import pandas as pd
<<<<<<< HEAD
from quapy.protocol import AbstractProtocol

class IFCBTrainSamplesFromDir(AbstractProtocol):

    def __init__(self, path_dir:str, classes: list):
        self.path_dir = path_dir
        self.classes = classes
        self.samples = []
        for filename in os.listdir(path_dir):
            if filename.endswith('.csv'):
                self.samples.append(filename)
=======
import math
from quapy.protocol import AbstractProtocol
from pathlib import Path

def get_sample_list(path_dir):
    """Gets a sample list finding the csv files in a directory

    Args:
        path_dir (_type_): directory to look for samples

    Returns:
        _type_: list of samples
    """
    samples = []
    for filename in sorted(os.listdir(path_dir)):
        if filename.endswith('.csv'):
            samples.append(filename)
    return samples

def generate_modelselection_split(samples, split=0.3):
    """This function generates a train/test split for model selection
    without the use of random numbers so the split is always the same

    Args:
        samples (_type_): list of samples
        split (float, optional): percentage saved for test. Defaults to 0.3.

    Returns:
        _type_: list of samples to use as train and list of samples to use as test
    """
    num_items_to_pick = math.ceil(len(samples) * split)
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
>>>>>>> 5566e0c97ae1b49b30874b6610d7f5b062009271

    def __call__(self):
        for sample in self.samples:
            s = pd.read_csv(os.path.join(self.path_dir,sample))
            # all columns but the first where we get the class
            X = s.iloc[:, 1:].to_numpy()
            y = s.iloc[:, 0].to_numpy()
            yield X, y

    def total(self):
        """
        Returns the total number of samples that the protocol generates.

        :return: The number of training samples to generate.
        """
        return len(self.samples)

<<<<<<< HEAD

class IFCBTestSamples(AbstractProtocol):

    def __init__(self, path_dir:str, test_prevalences_path: str):
        self.path_dir = path_dir
        self.test_prevalences = pd.read_csv(os.path.join(path_dir, test_prevalences_path))

    def __call__(self):
        for _, test_sample in self.test_prevalences.iterrows():
            #Load the sample from disk
            X = pd.read_csv(os.path.join(self.path_dir,test_sample['sample']+'.csv')).to_numpy()
            prevalences = test_sample.iloc[1:].to_numpy().astype(float)
=======
class IFCBTestSamples(AbstractProtocol):

    def __init__(self, path_dir:str, test_prevalences: pd.DataFrame, samples: list = None, classes: list=None):
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
>>>>>>> 5566e0c97ae1b49b30874b6610d7f5b062009271
            yield X, prevalences

    def total(self):
        """
        Returns the total number of samples that the protocol generates.

<<<<<<< HEAD
        :return: The number of test samples to generate.
        """
        return len(self.test_prevalences.index)
=======
        :return: The number of training samples to generate.
        """
        return len(self.samples)
>>>>>>> 5566e0c97ae1b49b30874b6610d7f5b062009271
