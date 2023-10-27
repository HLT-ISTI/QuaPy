import os
import pandas as pd
import numpy as np
from quapy.protocol import AbstractProtocol

class IFCBSamplesFromDir(AbstractProtocol):

    def __init__(self, path_dir:str, classes: list, train=True):
        self.path_dir = path_dir
        self.train = train
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