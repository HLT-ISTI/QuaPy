import argparse
import quapy as qp
from data import ResultSubmission, evaluate_submission
import constants
import os
import pickle
from tqdm import tqdm
from data import gen_load_samples_T1, load_category_map
from glob import glob
import constants

"""
LeQua2022 prediction script 
"""

def main(args):

    # check the number of samples
    nsamples = len(glob(os.path.join(args.samples, '*.txt')))
    if nsamples not in {constants.DEV_SAMPLES, constants.TEST_SAMPLES}:
        print(f'Warning: The number of samples does neither coincide with the expected number of '
              f'dev samples ({constants.DEV_SAMPLES}) nor with the expected number of '
              f'test samples ({constants.TEST_SAMPLES}).')

    _, categories = load_category_map(args.catmap)

    # load pickled model
    model = pickle.load(open(args.model, 'rb'))

    # predictions
    predictions = ResultSubmission(categories=list(range(len(categories))))
    for sampleid, sample in tqdm(gen_load_samples_T1(args.samples, args.nf),
                                   desc='predicting', total=nsamples):
        predictions.add(sampleid, model.quantify(sample))

    # saving
    basedir = os.path.basename(args.output)
    if basedir:
        os.makedirs(basedir, exist_ok=True)
    predictions.dump(args.output)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='LeQua2022 prediction script')
    parser.add_argument('model', metavar='MODEL-PATH', type=str,
                        help='Path of saved model')
    parser.add_argument('samples', metavar='SAMPLES-PATH', type=str,
                        help='Path to the directory containing the samples')
    parser.add_argument('output', metavar='PREDICTIONS-PATH', type=str,
                        help='Path where to store the predictions file')
    parser.add_argument('catmap', metavar='CATEGORY-MAP-PATH', type=str,
                        help='Path to the category map file')
    parser.add_argument('nf', metavar='NUM-FEATURES', type=int,
                        help='Number of features seen during training')
    args = parser.parse_args()

    if not os.path.exists(args.samples):
        raise FileNotFoundError(f'path {args.samples} does not exist')
    if not os.path.isdir(args.samples):
        raise ValueError(f'path {args.samples} is not a valid directory')

    main(args)
