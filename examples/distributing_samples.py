"""
Imagine we want to generate many samples out of a collection, that we want to distribute for others to run their
own experiments in the very same test samples. One naive solution would come down to applying a given protocol to
our collection (say the artificial prevalence protocol on the 'academic-success' UCI dataset), store all those samples
on disk and make them available online. Distributing many such samples is undesirable.
In this example, we generate the indexes that allow anyone to regenerate the samples out of the original collection.
"""

import quapy as qp
from quapy.method.aggregative import PACC
from quapy.protocol import UPP

data = qp.datasets.fetch_UCIMulticlassDataset('academic-success')
train, test = data.train_test

# let us train a quantifier to check whether we can actually replicate the results
quantifier = PACC()
quantifier.fit(train)

# let us simulate our experimental results
protocol = UPP(test, sample_size=100, repeats=100, random_state=0)
our_mae = qp.evaluation.evaluate(quantifier, protocol=protocol, error_metric='mae')

print(f'We have obtained a MAE={our_mae:.3f}')

# let us distribute the indexes; we specify that we want the indexes, not the samples
protocol = UPP(test, sample_size=100, repeats=100, random_state=0, return_type='index')
indexes = protocol.samples_parameters()

# Imagine we distribute the indexes; now we show how to replicate our experiments.
from quapy.protocol import ProtocolFromIndex
data = qp.datasets.fetch_UCIMulticlassDataset('academic-success')
train, test = data.train_test
protocol = ProtocolFromIndex(data=test, indexes=indexes)
their_mae = qp.evaluation.evaluate(quantifier, protocol=protocol, error_metric='mae')

print(f'Another lab obtains a MAE={our_mae:.3f}')

