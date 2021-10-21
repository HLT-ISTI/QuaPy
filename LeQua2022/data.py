import quapy as qp
import numpy as np
import sklearn


# def load_binary_raw_document(path):
#     documents, labels = qp.data.from_text(path, verbose=0, class2int=True)
#     labels = np.asarray(labels)
#     labels[np.logical_or(labels == 1, labels == 2)] = 0
#     labels[np.logical_or(labels == 4, labels == 5)] = 1
#     return documents, labels


# def load_multiclass_raw_document(path):
#     return qp.data.from_text(path, verbose=0, class2int=False)


def load_binary_vectors(path, nF=None):
    return sklearn.datasets.load_svmlight_file(path, n_features=nF)


def gen_load_samples_T1A(path_dir:str, ground_truth_path:str = None):
    # for ... : yield
    pass


def gen_load_samples_T1B(path_dir:str, ground_truth_path:str = None):
    # for ... : yield
    pass


def gen_load_samples_T2A(path_dir:str, ground_truth_path:str = None):
    # for ... : yield
    pass


def gen_load_samples_T2B(path_dir:str, ground_truth_path:str = None):
    # for ... : yield
    pass


class ResultSubmission:
    def __init__(self, team_name, run_name, task_name):
        assert isinstance(team_name, str) and team_name, \
            f'invalid value encountered for team_name'
        assert isinstance(run_name, str) and run_name, \
            f'invalid value encountered for run_name'
        assert isinstance(task_name, str) and task_name in {'T1A', 'T1B', 'T2A', 'T2B'}, \
            f'invalid value encountered for task_name; valid values are T1A, T1B, T2A, and T2B'
        self.team_name = team_name
        self.run_name = run_name
        self.task_name = task_name
        self.data = {}

    def add(self, sample_name:str, prevalence_values:np.ndarray):
        # assert the result is a valid sample_name (not repeated)
        pass

    def __len__(self):
        return len(self.data)

    @classmethod
    def load(cls, path:str)-> 'ResultSubmission':
        pass

    def dump(self, path:str):
        # assert all samples are covered (check for test and dev accordingly)
        pass

    def get(self, sample_name:str):
        pass


def evaluate_submission(ground_truth_prevs: ResultSubmission, submission_prevs: ResultSubmission):

    pass





