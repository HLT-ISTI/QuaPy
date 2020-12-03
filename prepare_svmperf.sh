#!/bin/bash
set -x

URL=http://download.joachims.org/svm_perf/current/svm_perf.tar.gz
FILE=./svm_perf.tar.gz
wget $URL $FILE
mkdir ./svm_perf
tar xvzf $FILE -C ./svm_perf
rm $FILE

#para crear el patch [para mi]
#diff -Naur svm_perf svm_perf_quantification > svm-perf-quantification-ext.patch

#para crear la modificacion
#cp svm_perf svm_perf_quantification -r [ESTO NO HACE FALTA]
patch -s -p0 < svm-perf-quantification-ext.patch
mv svm_perf svm_perf_quantification
cd svm_perf_quantification
make









