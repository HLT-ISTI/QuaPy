import numpy as np
from sklearn.linear_model import LogisticRegression
import quapy as qp
import quapy.functional as F
from quapy.data.datasets import LEQUA2022_SAMPLE_SIZE, fetch_lequa2022
from quapy.evaluation import evaluation_report
from quapy.method.aggregative import EMQ
from quapy.model_selection import GridSearchQ
import pandas as pd

"""
This example shows hoy to use the LeQua datasets (new in v0.1.7). For more information about the datasets, and the
LeQua competition itself, check:
https://lequa2022.github.io/index (the site of the competition)
https://ceur-ws.org/Vol-3180/paper-146.pdf (the overview paper)
"""

# there are 4 tasks (T1A, T1B, T2A, T2B), let us symply consider T1A (binary quantification, vector form)
task = 'T1A'

# set the sample size in the environment. The sample size is task-dendendent and can be consulted by doing:
qp.environ['SAMPLE_SIZE'] = LEQUA2022_SAMPLE_SIZE[task]
qp.environ['N_JOBS'] = -1

# the fetch method returns a training set (an instance of LabelledCollection) and two generators: one for the
# validation set and another for the test sets. These generators are both instances of classes that extend
# AbstractProtocol (i.e., classes that implement sampling generation procedures) and, in particular, are instances
# of SamplesFromDir, a protocol that simply iterates over pre-generated samples (those provided for the competition)
# stored in a directory.
training, val_generator, test_generator = fetch_lequa2022(task=task)
Xtr, ytr = training.Xy

# define the quantifier
quantifier = EMQ(classifier=LogisticRegression())

# model selection
param_grid = {
    'classifier__C': np.logspace(-3, 3, 7),          # classifier-dependent: inverse of regularization strength
    'classifier__class_weight': ['balanced', None],  # classifier-dependent: weights of each class
    # 'calib': ['bcts', None]                 # quantifier-dependent: recalibration method (new in v0.1.7)
}
model_selection = GridSearchQ(quantifier, param_grid, protocol=val_generator, error='mrae', refit=False, verbose=True)
quantifier = model_selection.fit(Xtr, ytr)

# evaluation
report = evaluation_report(quantifier, protocol=test_generator, error_metrics=['mae', 'mrae', 'mkld'], verbose=True)

# printing results
pd.set_option('display.expand_frame_repr', False)
report['estim-prev'] = report['estim-prev'].map(F.strprev)
print(report)

print('Averaged values:')
print(report.mean(numeric_only=True))
