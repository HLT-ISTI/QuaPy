#!/bin/bash
set -x

# T1A: vector-binary
# T1B: vector-multiclass
# T2A: raw-binary
# T2B: raw-multiclass


# finetuning
python3 finetune_bert.py ./data/T2A.train_dev/T2A/public/training_data.txt --modelname roberta-finetuned-T2A
python3 finetune_bert.py ./data/T2B.train_dev/T2B/public/training_data.txt --modelname roberta-finetuned-T2B

# dev
# ------
# extracting the embeddings
python3 generate_bert_embeddings.py roberta-finetuned-T2A/best_checkpoint ./data/T2A.train_dev/T2A ./data/T2A.train_dev_BERT/T2A T2A
python3 generate_bert_embeddings.py roberta-finetuned-T2B/best_checkpoint ./data/T2B.train_dev/T2B ./data/T2B.train_dev_BERT/T2B T2B





