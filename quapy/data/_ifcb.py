import os
import pandas as pd
from quapy.protocol import AbstractProtocol

class IFCBTrainSamplesFromDir(AbstractProtocol):

    def __init__(self, path_dir:str, classes: list):
        self.path_dir = path_dir
        self.classes = classes
        self.samples = []
        for filename in os.listdir(path_dir):
            if filename.endswith('.csv'):
                self.samples.append(filename)

    def __call__(self):
        for sample in self.samples:
            s = pd.read_csv(os.path.join(self.path_dir,sample))
            # all columns but the first where we get the class
            X = s.iloc[:, 1:].to_numpy()
            y = s.iloc[:, 0].to_numpy()
            yield X, y

class IFCBTestSamples(AbstractProtocol):

    def __init__(self, path_dir:str, test_prevalences_path: str):
        self.path_dir = path_dir
        self.test_prevalences = pd.read_csv(os.path.join(path_dir, test_prevalences_path))

    def __call__(self):
        for _, test_sample in self.test_prevalences.iterrows():
            #Load the sample from disk
            X = pd.read_csv(os.path.join(self.path_dir,test_sample['sample']+'.csv')).to_numpy()
            prevalences = test_sample.iloc[1:].to_numpy().astype(float)
            yield X, prevalences