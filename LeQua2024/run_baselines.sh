#!/bin/bash
set -x


# T1: binary (n=2)
# T2: multiclass (n=28)
# T3: ordinal (n=5)
# T4: covariante shift (n=2)

# preparing the environment: downloads the official LeQua 2024 scripts (only once and for all)
SCRIPTS_URL=https://github.com/HLT-ISTI/LeQua2024_scripts/archive/refs/heads/main.zip

# download and unzip the LeQua 2024 scripts

if [ ! -d "./scripts" ]; then
    echo "Downloading $SCRIPTS_URL into ./scripts"
    wget -qO scripts.zip "$SCRIPTS_URL"
    unzip -q scripts.zip
    mv "LeQua2024_scripts-main" "scripts"
    rm scripts.zip
    echo "[Done]"
else
    echo "LeQua 2024 scripts already exists"
fi


for task in T1 T2 T3 T4 ; do

  PYTHONPATH=.:scripts/:.. python3 baselines.py $task data/

  TEST_SAMPLES=data/lequa2024/$task/public/test_samples

  for pickledmodel in models/$task/*.pkl ; do
    model=$(basename "$pickledmodel" .pkl)
    PREDICTIONS=predictions/$model/task_"${task: -1}".csv
    PYTHONPATH=.:scripts/:.. python3 predict.py models/$task/$model.pkl $TEST_SAMPLES $PREDICTIONS
  done

done

echo "generating submission files for codalab in folder ./submission_files"

mkdir -p submission_files

for modelname in predictions/* ; do
  modelname=$(basename "$modelname")
  submission_name=submission_files/$modelname.zip
  rm -f $submission_name
  echo "zipping results for $modelname"
  zip -j $submission_name predictions/$modelname/task_*.csv
done

echo "[Done]"
