#!/bin/bash
set -x

URL=http://download.joachims.org/svm_perf/current/svm_perf.tar.gz
FILE=./svm_perf.tar.gz
wget $URL $FILE
mkdir ./svm_perf
tar xvzf $FILE -C ./svm_perf
rm $FILE

patch -s -p0 < svm-perf-quantification-ext.patch
mv svm_perf svm_perf_quantification
cd svm_perf_quantification
make









