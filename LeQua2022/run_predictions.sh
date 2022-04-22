#!/bin/bash
set -x


# T1A: vector-binary
# T1B: vector-multiclass
# T2A: raw-binary
# T2B: raw-multiclass


# extracting the embeddings 
echo "generating bert embeddings out of plain text"
python3 generate_bert_embeddings.py roberta-finetuned-T2A/best_checkpoint ./data/T2A/public ./data/T2A.test_BERT/public T2A
python3 generate_bert_embeddings.py roberta-finetuned-T2B/best_checkpoint ./data/T2B/public ./data/T2B.test_BERT/public T2B

echo "pickling numeric txt files as npy pickles for faster reuse"
PYTHONPATH=.:.. python3 prepickle_folder.py ./data/T1A/public ./data/T1A.test_pkl/public
PYTHONPATH=.:.. python3 prepickle_folder.py ./data/T1B/public ./data/T1B.test_pkl/public
PYTHONPATH=.:.. python3 prepickle_folder.py ./data/T2A.test_BERT ./data/T2A.test_BERT_pkl
PYTHONPATH=.:.. python3 prepickle_folder.py ./data/T2B.test_BERT ./data/T2B.test_BERT_pkl

echo "redefining sources"
T1Asamples=./data/T1A.test_pkl/public/test_samples/
T1Bsamples=./data/T1B.test_pkl/public/test_samples/
T2Asamples=./data/T2A.test_BERT_pkl/public/test_samples/
T2Bsamples=./data/T2B.test_BERT_pkl/public/test_samples/
T2Arawsamples=./data/T2A/public/test_samples/
T2Brawsamples=./data/T2B/public/test_samples/

for task in A B ; do
    T=T1"$task"_npy
    samples=./data/T1$task.test_pkl/public/test_samples/
    trueprev=./data/T1$task/private/test_prevalences.txt
    for Q in model/$T/*.pkl ; do 
        Q="$(basename -- $Q)"
        Q=${Q%.pkl}
        PYTHONPATH=.:.. python3 predict.py model/$T/$Q.pkl $samples predictions/$T/$Q.csv
        PYTHONPATH=.:.. python3 evaluate.py T1$task $trueprev ./predictions/$T/$Q.csv --output scores/$T/$Q.txt
    done
done


for task in A B ; do
    T=T2"$task"_BERT_npy
    samples=./data/T2$task.test_BERT_pkl/public/test_samples/
    trueprev=./data/T2$task/private/test_prevalences.txt
    for Q in model/$T/*.pkl ; do 
        Q="$(basename -- $Q)"
        Q=${Q%.pkl}
        PYTHONPATH=.:.. python3 predict.py model/$T/$Q.pkl $samples predictions/$T/$Q.csv
        PYTHONPATH=.:.. python3 evaluate.py T1$task $trueprev ./predictions/$T/$Q.csv --output scores/$T/$Q.txt
    done
done


for task in A B ; do
    T=T2"$task"_tfidf_raw
    samples=./data/T2$task/public/test_samples/
    trueprev=./data/T2$task/private/test_prevalences.txt
    for Q in model/$T/*.pkl ; do 
        Q="$(basename -- $Q)"
        Q=${Q%.pkl}
        PYTHONPATH=.:.. python3 predict.py model/$T/$Q.pkl $samples predictions/$T/$Q.csv --format raw
        PYTHONPATH=.:.. python3 evaluate.py T1$task $trueprev ./predictions/$T/$Q.csv --output scores/$T/$Q.txt
    done
done


