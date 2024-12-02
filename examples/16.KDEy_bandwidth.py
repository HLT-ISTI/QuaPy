import quapy as qp
import numpy as np
from quapy.protocol import UPP
from quapy.method.aggregative import KDEyML
import quapy.functional as F
from time import time

"""
Let see one example:  
"""

# load some data
qp.environ['SAMPLE_SIZE'] = 100
data = qp.datasets.fetch_UCIMulticlassDataset('molecular')
training, test = data.train_test
training, validation = training.split_stratified(train_prop=0.7, random_state=0)
protocol = UPP(validation)

hyper_C = np.logspace(-3, 3, 7)

model = KDEyML()

with qp.util.temp_seed(0):

    param_grid = {
        'classifier__C': hyper_C,
        'bandwidth': np.linspace(0.01, 0.20, 20) # [0.01, 0.02, 0.03, ..., 0.20]
    }

    model = qp.model_selection.GridSearchQ(
        model=model,
        param_grid=param_grid,
        protocol=protocol,
        error='mae',  # the error to optimize is the MAE (a quantification-oriented loss)
        refit=False,  # retrain on the whole labelled set once done
        n_jobs=-1,
        verbose=True  # show information as the process goes on
    ).fit(training)

best_params = model.best_params_
took = model.fit_time_
model = model.best_model_
print(f'model selection ended: best hyper-parameters={best_params}')

# evaluation in terms of MAE
# we use the same evaluation protocol (APP) on the test set
mae_score = qp.evaluation.evaluate(model, protocol=UPP(test), error_metric='mae')

print(f'MAE={mae_score:.5f}')
print(f'model selection took {took:.1f}s')


model = KDEyML(bandwidth='auto')

with qp.util.temp_seed(0):

    param_grid = {
        'classifier__C': hyper_C,
    }

    model = qp.model_selection.GridSearchQ(
        model=model,
        param_grid=param_grid,
        protocol=protocol,
        error='mae',  # the error to optimize is the MAE (a quantification-oriented loss)
        refit=False,  # retrain on the whole labelled set once done
        n_jobs=-1,
        verbose=True  # show information as the process goes on
    ).fit(training)

best_params = model.best_params_
took = model.fit_time_
model = model.best_model_
bandwidth = model.bandwidth_val
print(f'model selection ended: best hyper-parameters={best_params} ({bandwidth=})')

# evaluation in terms of MAE
# we use the same evaluation protocol (APP) on the test set
mae_score = qp.evaluation.evaluate(model, protocol=UPP(test), error_metric='mae')

print(f'MAE={mae_score:.5f}')
print(f'model selection took {took:.1f}s')

