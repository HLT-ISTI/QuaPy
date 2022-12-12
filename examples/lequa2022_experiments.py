import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import quapy as qp
import quapy.functional as F
from data.datasets import LEQUA2022_SAMPLE_SIZE, fetch_lequa2022
from evaluation import evaluation_report
from method.aggregative import EMQ
from model_selection import GridSearchQ
import pandas as pd


task = 'T1A'

qp.environ['SAMPLE_SIZE'] = LEQUA2022_SAMPLE_SIZE[task]
training, val_generator, test_generator = fetch_lequa2022(task=task)

# define the quantifier
learner = CalibratedClassifierCV(LogisticRegression())
quantifier = EMQ(learner=learner)

# model selection
param_grid = {'C': np.logspace(-3, 3, 7), 'class_weight': ['balanced', None]}
model_selection = GridSearchQ(quantifier, param_grid, protocol=val_generator, n_jobs=-1, refit=False, verbose=True)
quantifier = model_selection.fit(training)

# evaluation
report = evaluation_report(quantifier, protocol=test_generator, error_metrics=['mae', 'mrae', 'mkld'], verbose=True)

# printing results
pd.set_option('display.expand_frame_repr', False)
report['estim-prev'] = report['estim-prev'].map(F.strprev)
print(report)

print('Averaged values:')
print(report.mean())
