import argparse
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LR
from LeQua2022.methods import TfidfPipeline
from LeQua2022.quanet_lequa import QuaNetTrainerLeQua2022
from quapy.method.aggregative import *
from quapy.method.non_aggregative import MaximumLikelihoodPrevalenceEstimation as MLPE
import quapy.functional as F
from data import *
import os


# LeQua official baselines for LeQua2022
# =================================================================================

def baselines():
    yield CC(LR(n_jobs=-1)), "CC"
    yield ACC(LR(n_jobs=-1)), "ACC"
    yield PCC(LR(n_jobs=-1)), "PCC"
    yield PACC(LR(n_jobs=-1)), "PACC"
    yield EMQ(CalibratedClassifierCV(LR(), n_jobs=-1)), "SLD"
    yield HDy(LR(n_jobs=-1)) if args.task == 'binary' else OneVsAll(HDy(LR()), n_jobs=-1), "HDy"
    yield MLPE(), "MLPE"


def main(args):

    models_path = qp.util.create_if_not_exist(args.modeldir)

    path_dev_vectors = os.path.join(args.datadir, 'dev_samples')
    path_dev_prevs = os.path.join(args.datadir, 'dev_prevalences.txt')
    path_train = os.path.join(args.datadir, 'training_data.txt')

    qp.environ['SAMPLE_SIZE'] = constants.SAMPLE_SIZE[args.task]

    tfidfwrap = False
    if args.format == 'vector':
        load_fn, loader_kwargs = load_vector_documents, {}
    elif args.format == 'raw':
        load_fn, loader_kwargs = load_raw_documents, {}
        tfidfwrap = True
    else:
        load_fn, loader_kwargs = load_npy_documents, {'labeled':True}

    train = LabelledCollection.load(path_train, load_fn, **loader_kwargs)
    def gen_samples():
        return gen_load_samples(path_dev_vectors, path_dev_prevs, load_fn=load_fn)

    print(f'number of classes: {len(train.classes_)}')
    print(f'number of training documents: {len(train)}')
    print(f'training prevalence: {F.strprev(train.prevalence())}')
    print(f'training matrix shape: {train.instances.shape}')

    param_grid = {
        'C': np.logspace(-3, 3, 7),
        'class_weight': ['balanced', None]
    }

    for quantifier, q_name in baselines():
        quantifier = TfidfPipeline(quantifier) if tfidfwrap else quantifier
        print(f'{q_name}: Model selection')
        quantifier = qp.model_selection.GridSearchQ(
            quantifier,
            param_grid,
            sample_size=None,
            protocol='gen',
            error=qp.error.mrae,
            refit=False,
            verbose=True
        ).fit(train, gen_samples)

        print(f'{q_name} got MRAE={quantifier.best_score_:.5f} (hyper-params: {quantifier.best_params_})')

        model_path = os.path.join(models_path, q_name+'.pkl')
        print(f'saving model in {model_path}')
        pickle.dump(quantifier.best_model(), open(model_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LeQua2022 baselines')
    parser.add_argument('task', metavar='TASK', type=str, choices=['binary', 'multiclass'],
                        help='Task type (binary, multiclass)')
    parser.add_argument('datadir', metavar='DATA-PATH', type=str,
                        help='Path of the directory containing "dev_prevalences.txt", "training_data.txt|.npy", and '
                             'the directory "dev_samples"')
    parser.add_argument('modeldir', metavar='MODEL-PATH', type=str,
                        help='Path where to save the models.')
    parser.add_argument('format', metavar='FORMAT', type=str, choices=['raw', 'vector', 'npy'],
                        help='Format in which the samples are represented. '
                             'Use "raw" for textual documents, or '
                             '"vector" for vectorial representations (dumped as .txt files), or'
                             '"npy" for vectorial representation (dumped as .npy files --see prepickle_folder.py).')
    args = parser.parse_args()

    if not os.path.exists(args.datadir):
        raise FileNotFoundError(f'path {args.datadir} does not exist')
    if not os.path.isdir(args.datadir):
        raise ValueError(f'path {args.datadir} is not a valid directory')
    if not os.path.exists(os.path.join(args.datadir, "dev_prevalences.txt")):
        raise FileNotFoundError(f'path {args.datadir} does not contain "dev_prevalences.txt" file')
    if args.format == 'npy':
        if not os.path.exists(os.path.join(args.datadir, "training_data.npy")):
            raise FileNotFoundError(f'path {args.datadir} does not contain "training_data.npy" file')
    else:
        if not os.path.exists(os.path.join(args.datadir, "training_data.txt")):
            raise FileNotFoundError(f'path {args.datadir} does not contain "training_data.txt" file')
    if not os.path.exists(os.path.join(args.datadir, "dev_samples")):
        raise FileNotFoundError(f'path {args.datadir} does not contain "dev_samples" folder')

    main(args)
