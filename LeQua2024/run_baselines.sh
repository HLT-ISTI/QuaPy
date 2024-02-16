#!/bin/bash
set -x

# download the official scripts 
if [ ! -d "scripts" ]; then
   echo "Downloading the official scripts from the LeQua 2024 github repo"
   wget https://github.com/HLT-ISTI/LeQua2024_scripts/archive/refs/heads/main.zip
   unzip main.zip 
   mv LeQua2024_scripts-main scripts
   rm main.zip
fi

# T1: binary (n=2)
# T2: multiclass (n=28)
# T3: ordinal (n=5)
# T4: covariante shift (n=2)

# --------------------------------------------------------------------------------
# DEV
# --------------------------------------------------------------------------------

mkdir results

for task in T1 T2 T3 T4 ; do

  echo "" > results/$task.txt

  PYTHONPATH=.:scripts/:.. python3 baselines.py $task data/

  SAMPLES=data/$task/public/dev_samples
  TRUEPREVS=data/$task/public/dev_prevalences.txt

  for pickledmodel in models/$task/*.pkl ; do
    model=$(basename "$pickledmodel" .pkl)

    PREDICTIONS=predictions/$task/$model.txt

    PYTHONPATH=.:scripts/:.. python3 predict.py models/$task/$model.pkl $SAMPLES $PREDICTIONS
    PYTHONPATH=.:scripts/:.. python3 scripts/evaluate.py $task $TRUEPREVS $PREDICTIONS >> results/$task.txt
  done

done

