import pandas as pd
import os
from os.path import join

from scripts.data import load_vector_documents

from quapy.data import LabelledCollection
from quapy.protocol import AbstractProtocol


LEQUA2024_TASKS = ['T1', 'T2', 'T3', 'T4']


class LabelledCollectionsFromDir(AbstractProtocol):

    def __init__(self, path_dir:str, ground_truth_path:str, load_fn):
        self.path_dir = path_dir
        self.load_fn = load_fn
        self.true_prevs = pd.read_csv(ground_truth_path, index_col=0)

    def __call__(self):
        for id, prevalence in self.true_prevs.iterrows():
            collection_path = os.path.join(self.path_dir, f'{id}.txt')
            lc = LabelledCollection.load(path=collection_path, loader_func=self.load_fn)
            yield lc


def fetch_lequa2024(task, data_home='./data', merge_T3=False):

    from quapy.data._lequa2022 import SamplesFromDir

    assert task in LEQUA2024_TASKS, \
        f'Unknown task {task}. Valid ones are {LEQUA2024_TASKS}'

    # if data_home is None:
    #     data_home = get_quapy_home()
    lequa_dir = data_home

    # URL_TRAINDEV=f'https://zenodo.org/record/6546188/files/{task}.train_dev.zip'
    # URL_TEST=f'https://zenodo.org/record/6546188/files/{task}.test.zip'
    # URL_TEST_PREV=f'https://zenodo.org/record/6546188/files/{task}.test_prevalences.zip'

    # lequa_dir = join(data_home, 'lequa2024')
    # os.makedirs(lequa_dir, exist_ok=True)

    # def download_unzip_and_remove(unzipped_path, url):
    #     tmp_path = join(lequa_dir, task + '_tmp.zip')
    #     download_file_if_not_exists(url, tmp_path)
    #     with zipfile.ZipFile(tmp_path) as file:
    #         file.extractall(unzipped_path)
    #     os.remove(tmp_path)

    # if not os.path.exists(join(lequa_dir, task)):
    #     download_unzip_and_remove(lequa_dir, URL_TRAINDEV)
    #     download_unzip_and_remove(lequa_dir, URL_TEST)
    #     download_unzip_and_remove(lequa_dir, URL_TEST_PREV)

    load_fn = load_vector_documents

    val_samples_path = join(lequa_dir, task, 'public', 'dev_samples')
    val_true_prev_path = join(lequa_dir, task, 'public', 'dev_prevalences.txt')
    val_gen = SamplesFromDir(val_samples_path, val_true_prev_path, load_fn=load_fn)

    # test_samples_path = join(lequa_dir, task, 'public', 'test_samples')
    # test_true_prev_path = join(lequa_dir, task, 'public', 'test_prevalences.txt')
    # test_gen = SamplesFromDir(test_samples_path, test_true_prev_path, load_fn=load_fn)
    test_gen = None

    if task != 'T3':
        tr_path = join(lequa_dir, task, 'public', 'training_data.txt')
        train = LabelledCollection.load(tr_path, loader_func=load_fn)
        return train, val_gen, test_gen
    else:
        training_samples_path = join(lequa_dir, task, 'public', 'training_samples')
        training_true_prev_path = join(lequa_dir, task, 'public', 'training_prevalences.txt')
        train_gen = LabelledCollectionsFromDir(training_samples_path, training_true_prev_path, load_fn=load_fn)
        if merge_T3:
            train = LabelledCollection.join(*list(train_gen()))
            return train, val_gen, test_gen
        else:
            return train_gen, val_gen, test_gen

