import numpy as np
from abstention.calibration import NoBiasVectorScaling, VectorScaling, TempScaling
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import quapy as qp
import quapy.functional as F
from classification.calibration import RecalibratedProbabilisticClassifierBase, NBVSCalibration, \
    BCTSCalibration
from data.datasets import LEQUA2022_SAMPLE_SIZE, fetch_lequa2022
from evaluation import evaluation_report
from method.aggregative import EMQ
from model_selection import GridSearchQ
import pandas as pd

for task in ['T1A', 'T1B']:

        # calibration = TempScaling(verbose=False, bias_positions='all')

        qp.environ['SAMPLE_SIZE'] = LEQUA2022_SAMPLE_SIZE[task]
        training, val_generator, test_generator = fetch_lequa2022(task=task)

        # define the quantifier
        # learner = BCTSCalibration(LogisticRegression(), n_jobs=-1)
        # learner = CalibratedClassifierCV(LogisticRegression())
        learner = LogisticRegression()
        quantifier = EMQ(classifier=learner)

        # model selection
        param_grid = {
            'classifier__C': np.logspace(-3, 3, 7),
            'classifier__class_weight': ['balanced', None],
            'recalib': ['platt', 'ts', 'vs', 'nbvs', 'bcts', None],
            'exact_train_prev': [False, True]
        }
        model_selection = GridSearchQ(quantifier, param_grid, protocol=val_generator, error='mrae', n_jobs=-1, refit=False, verbose=True)
        quantifier = model_selection.fit(training)

        # evaluation
        report = evaluation_report(quantifier, protocol=test_generator, error_metrics=['mae', 'mrae', 'mkld'], verbose=True)

        # import os
        # os.makedirs(f'./out', exist_ok=True)
        # with open(f'./out/EMQ_{calib}_{task}.txt', 'wt') as foo:
        #     estim_prev = report['estim-prev'].values
        #     nclasses = len(estim_prev[0])
        #     foo.write(f'id,'+','.join([str(x) for x in range(nclasses)])+'\n')
        #     for id, prev in enumerate(estim_prev):
        #         foo.write(f'{id},'+','.join([f'{p:.5f}' for p in prev])+'\n')
        #
        # #os.makedirs(f'./errors/{task}', exist_ok=True)
        # with open(f'./out/EMQ_{calib}_{task}_errors.txt', 'wt') as foo:
        #     maes, mraes = report['mae'].values, report['mrae'].values
        #     foo.write(f'id,AE,RAE\n')
        #     for id, (ae_i, rae_i) in enumerate(zip(maes, mraes)):
        #         foo.write(f'{id},{ae_i:.5f},{rae_i:.5f}\n')

        # printing results
        pd.set_option('display.expand_frame_repr', False)
        report['estim-prev'] = report['estim-prev'].map(F.strprev)
        print(report)

        print('Averaged values:')
        print(report.mean())
