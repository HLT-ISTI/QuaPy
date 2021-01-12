from sklearn.linear_model import LogisticRegression
import quapy as qp
import quapy.functional as F
import numpy as np
import os
import sys
import pickle

qp.environ['SAMPLE_SIZE'] = 100
sample_size = qp.environ['SAMPLE_SIZE']



def evaluate_experiment(true_prevalences, estim_prevalences, n_repetitions=25):
    #n_classes = true_prevalences.shape[1]
    #true_ave = true_prevalences.reshape(-1, n_repetitions, n_classes).mean(axis=1)
    #estim_ave = estim_prevalences.reshape(-1, n_repetitions, n_classes).mean(axis=1)
    #estim_std = estim_prevalences.reshape(-1, n_repetitions, n_classes).std(axis=1)
    #print('\nTrueP->mean(Phat)(std(Phat))\n'+'='*22)
    #for true, estim, std in zip(true_ave, estim_ave, estim_std):
    #    str_estim = ', '.join([f'{mean:.3f}+-{std:.4f}' for mean, std in zip(estim, std)])
    #    print(f'{F.strprev(true)}->[{str_estim}]')

    print('\nEvaluation Metrics:\n'+'='*22)
    for eval_measure in [qp.error.mae, qp.error.mrae]:
        err = eval_measure(true_prevalences, estim_prevalences)
        print(f'\t{eval_measure.__name__}={err:.4f}')
    print()


def evaluate_method_point_test(method, test):
    estim_prev = method.quantify(test.instances)
    true_prev = F.prevalence_from_labels(test.labels, test.n_classes)
    print('\nPoint-Test evaluation:\n' + '=' * 22)
    print(f'true-prev={F.strprev(true_prev)}, estim-prev={F.strprev(estim_prev)}')
    for eval_measure in [qp.error.mae, qp.error.mrae]:
        err = eval_measure(true_prev, estim_prev)
        print(f'\t{eval_measure.__name__}={err:.4f}')


def quantification_models():
    def newLR():
        return LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
    __C_range = np.logspace(-4, 5, 10)
    lr_params = {'C': __C_range, 'class_weight': [None, 'balanced']}
    #yield 'cc', qp.method.aggregative.CC(newLR()), lr_params
    #yield 'acc', qp.method.aggregative.ACC(newLR()), lr_params
    #yield 'pcc', qp.method.aggregative.PCC(newLR()), lr_params
    yield 'pacc', qp.method.aggregative.PACC(newLR()), lr_params


def result_path(dataset_name, model_name, optim_metric):
    return f'{dataset_name}-{model_name}-{optim_metric}.pkl'


def check_already_computed(dataset_name, model_name, optim_metric):
    path = result_path(dataset_name, model_name, optim_metric)
    return os.path.exists(path)


def save_results(dataset_name, model_name, optim_metric, *results):
    path = result_path(dataset_name, model_name, optim_metric)
    qp.util.create_parent_dir(path)
    with open(path, 'wb') as foo:
        pickle.dump(tuple(results), foo, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    np.random.seed(0)

    for dataset_name in ['sanders']:  # qp.datasets.TWITTER_SENTIMENT_DATASETS:

        benchmark_devel = qp.datasets.fetch_twitter(dataset_name, for_model_selection=True, min_df=5, pickle=True)
        benchmark_devel.stats()

        for model_name, model, hyperparams in quantification_models():

            model_selection = qp.model_selection.GridSearchQ(
                model,
                param_grid=hyperparams,
                sample_size=sample_size,
                n_prevpoints=21,
                n_repetitions=5,
                error='mae',
                refit=False,
                verbose=True
            )

            model_selection.fit(benchmark_devel.training, benchmark_devel.test)
            model = model_selection.best_model()

            benchmark_eval = qp.datasets.fetch_twitter(dataset_name, for_model_selection=False, min_df=5, pickle=True)
            model.fit(benchmark_eval.training)
            true_prevalences, estim_prevalences = qp.evaluation.artificial_sampling_prediction(
                model,
                test=benchmark_eval.test,
                sample_size=sample_size,
                n_prevpoints=21,
                n_repetitions=25
            )

            evaluate_experiment(true_prevalences, estim_prevalences, n_repetitions=25)
            evaluate_method_point_test(model, benchmark_eval.test)

            #save_arrays(FLAGS.results, true_prevalences, estim_prevalences, test_name)

            sys.exit(0)

            # decide the test to be performed (in the case of 'semeval', tests are 'semeval13', 'semeval14', 'semeval15')
            if FLAGS.dataset == 'semeval':
                test_sets = ['semeval13', 'semeval14', 'semeval15']
            else:
                test_sets = [FLAGS.dataset]

                evaluate_method_point_test(method, benchmark_eval.test, test_name=test_set)




# quantifiers:
# ----------------------------------------
# alias for quantifiers and default configurations
QUANTIFIER_ALIASES = {
    'cc': lambda learner: ClassifyAndCount(learner),
    'acc': lambda learner: AdjustedClassifyAndCount(learner),
    'pcc': lambda learner: ProbabilisticClassifyAndCount(learner),
    'pacc': lambda learner: ProbabilisticAdjustedClassifyAndCount(learner),
    'emq': lambda learner: ExpectationMaximizationQuantifier(learner),
    'svmq': lambda learner: OneVsAllELM(settings.SVM_PERF_HOME, loss='q'),
    'svmkld': lambda learner: OneVsAllELM(settings.SVM_PERF_HOME, loss='kld'),
    'svmnkld': lambda learner: OneVsAllELM(settings.SVM_PERF_HOME, loss='nkld'),
    'svmmae': lambda learner: OneVsAllELM(settings.SVM_PERF_HOME, loss='mae'),
    'svmmrae': lambda learner: OneVsAllELM(settings.SVM_PERF_HOME, loss='mrae'),
    'mlpe': lambda learner: MaximumLikelihoodPrevalenceEstimation(),
}

