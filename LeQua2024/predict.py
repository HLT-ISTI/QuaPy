import argparse
import quapy as qp
from scripts.data import ResultSubmission
import os
import pickle
from tqdm import tqdm
from scripts.data import gen_load_samples
from glob import glob
from scripts import constants
from regressor import KDEyRegressor, RegressionToSimplex

"""
LeQua2024 prediction script 
"""

def main(args):

    if not args.force and os.path.exists(args.output):
        print(f'prediction file {args.output} already exists! set --force to override')
        return

    # check the number of samples
    nsamples = len(glob(os.path.join(args.samples, f'*.txt')))
    if nsamples not in {constants.DEV_SAMPLES, constants.TEST_SAMPLES}:
        print(f'Warning: The number of samples (.txt) in {args.samples} '
              f'does neither coincide with the expected number of '
              f'dev samples ({constants.DEV_SAMPLES}) nor with the expected number of '
              f'test samples ({constants.TEST_SAMPLES}).')

    # load pickled model
    model = pickle.load(open(args.model, 'rb'))

    # predictions
    predictions = ResultSubmission()
    for sampleid, sample in tqdm(gen_load_samples(args.samples, return_id=True), desc='predicting', total=nsamples):
        predictions.add(sampleid, model.quantify(sample))

    # saving
    qp.util.create_parent_dir(args.output)
    predictions.dump(args.output)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='LeQua2024 prediction script')
    parser.add_argument('model', metavar='MODEL-PATH', type=str,
                        help='Path of saved model')
    parser.add_argument('samples', metavar='SAMPLES-PATH', type=str,
                        help='Path to the directory containing the samples')
    parser.add_argument('output', metavar='PREDICTIONS-PATH', type=str,
                        help='Path where to store the predictions file')
    parser.add_argument('--force', action='store_true',
                        help='Overrides prediction file if exists')
    args = parser.parse_args()

    if not os.path.exists(args.samples):
        raise FileNotFoundError(f'path {args.samples} does not exist')
    if not os.path.isdir(args.samples):
        raise ValueError(f'path {args.samples} is not a valid directory')

    main(args)
