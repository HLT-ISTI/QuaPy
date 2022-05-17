import argparse
from sklearn.linear_model import LogisticRegression as LR

from data import load_npy_documents, ResultSubmission, gen_load_samples
from quanet_lequa import QuaNetTrainerLeQua2022
from quapy.method.aggregative import *
import quapy.functional as F
import os
import constants
from glob import glob


# LeQua official baselines for task T1A (Binary/Vector) and T1B (Multiclass/Vector)
# =================================================================================

def predict(path_samples, model, output):
    # check the number of samples
    nsamples = len(glob(os.path.join(path_samples, f'*.npy')))

    # predictions
    predictions = ResultSubmission()
    for sampleid, sample in tqdm(gen_load_samples(path_samples, return_id=True, ext='npy', load_fn=load_npy_documents),
                                 desc='predicting', total=nsamples):
        predictions.add(sampleid, model.quantify(sample))

    # saving
    qp.util.create_parent_dir(output)
    predictions.dump(output)


def main(args):

    path_dev_vectors = os.path.join(args.traindir, 'dev_samples')
    path_test_vectors = args.testdir
    path_dev_prevs = os.path.join(args.traindir, 'dev_prevalences.txt')
    path_train = os.path.join(args.traindir, 'training_data.txt')

    qp.environ['SAMPLE_SIZE'] = constants.SAMPLE_SIZE[args.task]

    train = LabelledCollection.load(path_train, load_npy_documents, labeled=True)

    print(f'number of classes: {len(train.classes_)}')
    print(f'number of training documents: {len(train)}')
    print(f'training prevalence: {F.strprev(train.prevalence())}')
    print(f'training matrix shape: {train.instances.shape}')

    quanet = QuaNetTrainerLeQua2022(
        learner=LR(),
        sample_size=constants.SAMPLE_SIZE[args.task],
        val_samples_dir=path_dev_vectors,
        groundtruth_path=path_dev_prevs,
        n_epochs=100,
        lr=1e-4,
        lstm_hidden_size=64,
        lstm_nlayers=1,
        ff_layers=[1024, 512],
        bidirectional=True,
        qdrop_p=0.5,
        patience=10,
        checkpointdir=args.checkpointdir,
        checkpointname=args.checkpointname,
        device='cuda'
    )

    quanet.fit(train)

    # we are not optimizing hyperparameters, so we could directly generate predictions for the test set
    # predictions
    predict(path_test_vectors, quanet, args.testoutput)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LeQua2022 baselines')
    parser.add_argument('task', metavar='TASK', type=str, choices=['binary', 'multiclass'],
                        help='Task type (binary, multiclass)')
    parser.add_argument('traindir', metavar='DATA-PATH', type=str,
                        help='Path of the directory containing "dev_prevalences.txt", "training_data.npy", and '
                             'the directory "dev_samples"')
    parser.add_argument('testdir', metavar='DATA-PATH', type=str,
                        help='Path of the directory of the "test_samples"')
    parser.add_argument('checkpointdir', metavar='MODEL-PATH', type=str,
                        help='Path where to save the model parameters.')
    parser.add_argument('devoutput', metavar='PREDICTIONS-PATH', type=str,
                        help='Path where to save the validation predictions.')
    parser.add_argument('testoutput', metavar='PREDICTIONS-PATH', type=str,
                        help='Path where to save the test predictions.')

    args = parser.parse_args()

    if not os.path.exists(args.traindir):
        raise FileNotFoundError(f'path {args.traindir} does not exist')
    if not os.path.isdir(args.traindir):
        raise ValueError(f'path {args.traindir} is not a valid directory')
    if not os.path.exists(os.path.join(args.traindir, "dev_prevalences.txt")):
        raise FileNotFoundError(f'path {args.traindir} does not contain "dev_prevalences.txt" file')
    if not os.path.exists(os.path.join(args.traindir, "dev_samples")):
        raise FileNotFoundError(f'path {args.traindir} does not contain "dev_samples" folder')
    if not os.path.exists(args.testdir):
        raise FileNotFoundError(f'path {args.testdir} does not exist')

    args.checkpointname = f'QuaNet_lr1e-4_h64_n1'

    main(args)
