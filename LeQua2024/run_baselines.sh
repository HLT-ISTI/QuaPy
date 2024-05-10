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

#  PYTHONPATH=.:scripts/:.. python3 baselines.py $task data/

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
  echo "modelname is $modelname"
  for taskname in predictions/$modelname/*.csv ; do
    taskname=$(basename "$taskname")
    submission_name=submission_files/"$modelname"_"$taskname".zip
    rm -f $submission_name
    echo "zipping results for $modelname and task $taskname"
    zip -j $submission_name predictions/$modelname/$taskname
  done
done

echo "[Done]"
