#!/bin/bash
set -x

#conda activate torch

transformer=roberta-base

#python3 finetune_bert.py ./data/Books/training_data.txt $transformer
#python3 generate_bert_vectors_npytxt.py "$transformer"-finetuned last
#python3 generate_bert_vectors_npytxt.py "$transformer"-finetuned average
PYTHONPATH=.:.. python3 generate_bert_vectors_npytxt.py "$transformer"-finetuned posteriors
