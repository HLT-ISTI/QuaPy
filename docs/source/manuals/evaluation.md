# Evaluation

Quantification is an appealing tool in scenarios of dataset shift, 
and particularly in scenarios of prior-probability shift. 
That is, the interest in estimating the class prevalences arises
under the belief that those class prevalences might have changed
with respect to the ones observed during training. 
In other words, one could simply return the training prevalence
as a predictor of the test prevalence if this change is assumed
to be unlikely (as is the case in general scenarios of 
machine learning governed by the iid assumption).
In brief, quantification requires dedicated evaluation protocols, 
which are implemented in QuaPy and explained here.

## Error Measures

The module quapy.error implements the most popular error measures for quantification, e.g., mean absolute error (_mae_), mean relative absolute error (_mrae_), among others. For each such measure (e.g., _mrae_) there are corresponding functions (e.g., _rae_) that do not average the results across samples.

Some errors of classification are also available, e.g., accuracy error (_acce_) or F-1 error (_f1e_).

The error functions implement the following interface, e.g.:

```python
mae(true_prevs, prevs_hat)
```

in which the first argument is a ndarray containing the true
prevalences, and the second argument is another ndarray with
the estimations produced by some method.

Some error functions, e.g., _mrae_, _mkld_, and _mnkld_, are 
smoothed for numerical stability. In those cases, there is a
third argument, e.g.:

```python
def mrae(true_prevs, prevs_hat, eps=None): ...
```

indicating the value for the smoothing parameter epsilon.
Traditionally, this value is set to 1/(2T) in past literature,
with T the sampling size. One could either pass this value
to the function each time, or to set a QuaPy's environment 
variable _SAMPLE_SIZE_ once, and omit this argument 
thereafter (recommended);
e.g.:

```python
qp.environ['SAMPLE_SIZE'] = 100  # once for all
true_prev = np.asarray([0.5, 0.3, 0.2])  # let's assume 3 classes
estim_prev = np.asarray([0.1, 0.3, 0.6])
error = qp.error.mrae(true_prev, estim_prev)
print(f'mrae({true_prev}, {estim_prev}) = {error:.3f}')
```

will print:
```
mrae([0.500, 0.300, 0.200], [0.100, 0.300, 0.600]) = 0.914
```

Finally, it is possible to instantiate QuaPy's quantification
error functions from strings using, e.g.:

```python
error_function = qp.error.from_name('mse')
error = error_function(true_prev, estim_prev)
```

## Evaluation Protocols

An _evaluation protocol_ is an evaluation procedure that uses
one specific _sample generation procotol_ to genereate many
samples, typically characterized by widely varying amounts of 
_shift_ with respect to the original distribution, that are then
used to evaluate the performance of a (trained) quantifier. 
These protocols are explained in more detail in a dedicated [entry 
in the wiki](Protocols.md). For the moment being, let us assume we already have
chosen and instantiated one specific such protocol, that we here
simply call _prot_. Let also assume our model is called
_quantifier_ and that our evaluatio measure of choice is 
_mae_. The evaluation comes down to:

```python
mae = qp.evaluation.evaluate(quantifier, protocol=prot, error_metric='mae')
print(f'MAE = {mae:.4f}')
```

It is often desirable to evaluate our system using more than one
single evaluatio measure. In this case, it is convenient to generate
a _report_. A report in QuaPy is a dataframe accounting for all the
true prevalence values with their corresponding prevalence values
as estimated by the quantifier, along with the error each has given
rise. 

```python
report = qp.evaluation.evaluation_report(quantifier, protocol=prot, error_metrics=['mae', 'mrae', 'mkld'])
```

From a pandas' dataframe, it is straightforward to visualize all the results, 
and compute the averaged values, e.g.: 

```python
pd.set_option('display.expand_frame_repr', False)
report['estim-prev'] = report['estim-prev'].map(F.strprev)
print(report)

print('Averaged values:')
print(report.mean())
```

This will produce an output like:

```
           true-prev      estim-prev       mae      mrae      mkld
0     [0.308, 0.692]  [0.314, 0.686]  0.005649  0.013182  0.000074
1     [0.896, 0.104]  [0.909, 0.091]  0.013145  0.069323  0.000985
2     [0.848, 0.152]  [0.809, 0.191]  0.039063  0.149806  0.005175
3     [0.016, 0.984]  [0.033, 0.967]  0.017236  0.487529  0.005298
4     [0.728, 0.272]  [0.751, 0.249]  0.022769  0.057146  0.001350
...              ...             ...       ...       ...       ...
4995    [0.72, 0.28]  [0.698, 0.302]  0.021752  0.053631  0.001133
4996  [0.868, 0.132]  [0.888, 0.112]  0.020490  0.088230  0.001985
4997  [0.292, 0.708]  [0.298, 0.702]  0.006149  0.014788  0.000090
4998    [0.24, 0.76]  [0.220, 0.780]  0.019950  0.054309  0.001127
4999  [0.948, 0.052]  [0.965, 0.035]  0.016941  0.165776  0.003538

[5000 rows x 5 columns]
Averaged values:
mae     0.023588
mrae    0.108779
mkld    0.003631
dtype: float64

Process finished with exit code 0
```

Alternatively, we can simply generate all the predictions by:

```python
true_prevs, estim_prevs = qp.evaluation.prediction(quantifier, protocol=prot)
```

All the evaluation functions implement specific optimizations for speeding-up 
the evaluation of aggregative quantifiers (i.e., of instances of _AggregativeQuantifier_).
The optimization comes down to generating classification predictions (either crisp or soft) 
only once for the entire test set, and then applying the sampling procedure to the
predictions, instead of generating samples of instances and then computing the 
classification predictions every time. This is only possible when the protocol
is an instance of _OnLabelledCollectionProtocol_. The optimization is only 
carried out when the number of classification predictions thus generated would be
smaller than the number of predictions required for the entire protocol; e.g., 
if the original dataset contains 1M instances, but the protocol is such that it would
at most generate 20 samples of 100 instances, then it would be preferable to postpone the
classification for each sample. This behaviour is indicated by setting 
_aggr_speedup="auto"_. Conversely, when indicating _aggr_speedup="force"_ QuaPy will
precompute all the predictions irrespectively of the number of instances and number of samples.
Finally, this can be deactivated by setting _aggr_speedup=False_. Note that this optimization
is not only applied for the final evaluation, but also for the internal evaluations carried
out during _model selection_. Since these are typically many, the heuristic can help reduce the
execution time a lot.