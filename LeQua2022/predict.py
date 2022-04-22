import argparse
import quapy as qp
from data import ResultSubmission, load_npy_documents, load_raw_documents
import os
import pickle
from tqdm import tqdm
from data import gen_load_samples
from glob import glob
import constants
from advanced_baselines import RegressionQuantification

"""
LeQua2022 prediction script 
"""

def main(args):

    if args.format == 'npy':
        ext='npy'
        loader_fn = load_npy_documents
    elif args.format == 'raw':
        ext='txt'
        loader_fn = load_raw_documents

    # check the number of samples
    nsamples = len(glob(os.path.join(args.samples, f'*.{ext}')))
    if nsamples not in {constants.DEV_SAMPLES, constants.TEST_SAMPLES}:
        print(f'Warning: The number of samples (.{ext}) in {args.samples} does neither coincide with the expected number of '
              f'dev samples ({constants.DEV_SAMPLES}) nor with the expected number of '
              f'test samples ({constants.TEST_SAMPLES}).')

    # load pickled model
    model = pickle.load(open(args.model, 'rb'))

    # predictions
    predictions = ResultSubmission()
    for sampleid, sample in tqdm(gen_load_samples(args.samples, return_id=True, ext=ext, load_fn=loader_fn), desc='predicting', total=nsamples):
        predictions.add(sampleid, model.quantify(sample))

    # saving
    qp.util.create_parent_dir(args.output)
    predictions.dump(args.output)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='LeQua2022 prediction script')
    parser.add_argument('model', metavar='MODEL-PATH', type=str,
                        help='Path of saved model')
    parser.add_argument('samples', metavar='SAMPLES-PATH', type=str,
                        help='Path to the directory containing the samples')
    parser.add_argument('output', metavar='PREDICTIONS-PATH', type=str,
                        help='Path where to store the predictions file')
    parser.add_argument('--format', metavar='FORMAT', type=str, choices=['npy', 'raw'],
                        help='format of data samples', default='npy')
    args = parser.parse_args()

    if not os.path.exists(args.samples):
        raise FileNotFoundError(f'path {args.samples} does not exist')
    if not os.path.isdir(args.samples):
        raise ValueError(f'path {args.samples} is not a valid directory')

    main(args)
