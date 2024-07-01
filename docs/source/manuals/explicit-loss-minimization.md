# Explicit Loss Minimization

QuaPy makes available several Explicit Loss Minimization (ELM) methods, including
SVM(Q), SVM(KLD), SVM(NKLD), SVM(AE), or SVM(RAE).
These methods require to first download the 
[svmperf](http://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html) 
package, apply the patch 
[svm-perf-quantification-ext.patch](https://github.com/HLT-ISTI/QuaPy/blob/master/svm-perf-quantification-ext.patch), and compile the sources.
The script [prepare_svmperf.sh](https://github.com/HLT-ISTI/QuaPy/blob/master/prepare_svmperf.sh) does all the job. Simply run:

```
./prepare_svmperf.sh
```

The resulting directory `svm_perf_quantification/` contains the
patched version of _svmperf_ with quantification-oriented losses. 

The [svm-perf-quantification-ext.patch](https://github.com/HLT-ISTI/QuaPy/blob/master/prepare_svmperf.sh) is an extension of the patch made available by
[Esuli et al. 2015](https://dl.acm.org/doi/abs/10.1145/2700406?casa_token=8D2fHsGCVn0AAAAA:ZfThYOvrzWxMGfZYlQW_y8Cagg-o_l6X_PcF09mdETQ4Tu7jK98mxFbGSXp9ZSO14JkUIYuDGFG0) 
that allows SVMperf to optimize for
the _Q_ measure as proposed by [Barranquero et al. 2015](https://www.sciencedirect.com/science/article/abs/pii/S003132031400291X) 
and for the _KLD_ and _NKLD_ measures as proposed by [Esuli et al. 2015](https://dl.acm.org/doi/abs/10.1145/2700406?casa_token=8D2fHsGCVn0AAAAA:ZfThYOvrzWxMGfZYlQW_y8Cagg-o_l6X_PcF09mdETQ4Tu7jK98mxFbGSXp9ZSO14JkUIYuDGFG0).
This patch extends the above one by also allowing SVMperf to optimize for 
_AE_ and _RAE_.
See the [](./methods) manual for more details and code examples.

