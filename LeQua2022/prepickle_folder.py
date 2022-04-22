import argparse
import numpy as np
from os.path import join
import os
import shutil
from tqdm import tqdm
from glob import glob
from pathlib import Path
import re
from data import load_vector_documents

"""
This scripts takes a task folder containing sample files in vector form (saved as .txt) and converts them into
ndarray vectors that are then efficiently stored for faster usage.
"""


def save_samples_as_npy(vectors, labels, path):
    if labels is not None:
        labels = labels.reshape(-1,1)
        vectors = np.hstack([labels, vectors])
    np.save(path, vectors)


def transform_folder(inputfolder, outputfolder):
    os.makedirs(outputfolder, exist_ok=True)
    files = list(glob(join(inputfolder, '**'), recursive=True))
    pbar = tqdm(files, total=len(files))
    for inpath in pbar:
        outpath = inpath.replace(inputfolder, outputfolder)
        os.makedirs(Path(outpath).parent, exist_ok=True)
        filename = Path(inpath).name
        if filename.endswith('.txt'):
            if filename == 'training_data.txt' or re.match(r"^\d+.txt$", filename):
                pbar.set_description(f'processing >> {inpath}')
                instances, labels = load_vector_documents(inpath)
                save_samples_as_npy(instances, labels, outpath.replace('.txt', '.npy'))
            else:
                pbar.set_description(f'copying {inpath} --> {outpath}')
                shutil.copyfile(inpath, outpath)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pre-pickles collections out of vector files')
    parser.add_argument('inputfolder', metavar='PATH', type=str,
                        help="Path to the folder containing the data samples in (.txt) vector format")
    parser.add_argument('outputfolder', metavar='PATH', type=str,
                        help="Path to the folder where to dump the pickled collections. "
                             "This folder will mimic the original file structure.")

    args = parser.parse_args()

    if not os.path.exists(args.inputfolder):
        raise FileNotFoundError(f'folder {args.inputfolder} does not exist')

    transform_folder(args.inputfolder, args.outputfolder)





