#!/bin/bash
set -x


# T1A: vector-binary
# T1B: vector-multiclass
# T2A: raw-binary
# T2B: raw-multiclass

# pre-pickle the vector representations (dumped as .txt files) as .npy bin files for faster reuse
# the directories involved are T1A and T1B (originally in "vector-txt" format) and the folders
# T2A and T2B that are generated via generate_bert_embeddings.py

# --------------------------------------------------------------------------------
# DEV
# --------------------------------------------------------------------------------

#PYTHONPATH=.:.. python3 prepickle_folder.py ./data/T1A.train_dev/T1A ./data/T1A.train_dev_pkl/T1A
#PYTHONPATH=.:.. python3 prepickle_folder.py ./data/T1B.train_dev/T1B ./data/T1B.train_dev_pkl/T1B
#PYTHONPATH=.:.. python3 prepickle_folder.py ./data/T2A.train_dev_BERT/T2A ./data/T2A.train_dev_BERT_pkl/T2A
#PYTHONPATH=.:.. python3 prepickle_folder.py ./data/T2B.train_dev_BERT/T2B ./data/T2B.train_dev_BERT_pkl/T2B

#PYTHONPATH=.:.. python3 baselines.py binary ./data/T1A.train_dev_pkl/T1A/public ./model/T1A_npy npy &
#PYTHONPATH=.:.. python3 baselines.py multiclass ./data/T1B.train_dev_pkl/T1B/public ./model/T1B_npy_mlpe npy
#PYTHONPATH=.:.. python3 baselines.py binary ./data/T2A.train_dev/T2A/public ./model/T2A_tfidf_raw raw &
#PYTHONPATH=.:.. python3 baselines.py multiclass ./data/T2B.train_dev/T2B/public ./model/T2B_tfidf_raw raw
#PYTHONPATH=.:.. python3 baselines.py binary ./data/T2A.train_dev_BERT_pkl/T2A/public ./model/T2A_BERT_npy npy &
#PYTHONPATH=.:.. python3 baselines.py multiclass ./data/T2B.train_dev_BERT_pkl/T2B/public ./model/T2B_BERT_npy npy &

# run other experimental baselines ...

#PYTHONPATH=.:.. python3 advanced_baselines.py binary ./data/T1A.train_dev_pkl/T1A/public ./model/T1A_npy npy 
#PYTHONPATH=.:.. python3 advanced_baselines.py multiclass ./data/T1B.train_dev_pkl/T1B/public ./model/T1B_npy npy 
#PYTHONPATH=.:.. python3 advanced_baselines.py binary ./data/T2A.train_dev/T2A/public ./model/T2A_tfidf_raw raw 
#PYTHONPATH=.:.. python3 advanced_baselines.py multiclass ./data/T2B.train_dev/T2B/public ./model/T2B_tfidf_raw raw 
#PYTHONPATH=.:.. python3 advanced_baselines.py binary ./data/T2A.train_dev_BERT_pkl/T2A/public ./model/T2A_BERT_npy npy 
#PYTHONPATH=.:.. python3 advanced_baselines.py multiclass ./data/T2B.train_dev_BERT_pkl/T2B/public ./model/T2B_BERT_npy npy 


#PYTHONPATH=.:.. python3 baseline_quanet.py binary ./data/T1A.train_dev_pkl/T1A/public ./data/T1A.test_pkl/public/test_samples quanet_model/T1A predictions/T1A/quanet_dev.csv predictions/T1A/quanet_test.csv
#PYTHONPATH=.:.. python3 baseline_quanet.py binary ./data/T2A.train_dev_BERT_pkl/T2A/public ./data/T2A.test_BERT_pkl/public/test_samples quanet_model/T2A predictions/T2A/quanet_dev.csv predictions/T2A/quanet_test.csv

