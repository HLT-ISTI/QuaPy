import argparse
import csv
import os
import sys
import datasets
import numpy as np
import pandas as pd
import torch.cuda
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments

from LeQua2022.utils import clean_checkpoints

"""
This script fine-tunes a pre-trained language model on a given textual training set.
The training goes for a maximum of 10 epochs, but stores the model parameters of the best performing epoch according
to the validation loss in a held-out val split of 1000 documents (stratified).

Example: to fine-tune RoBERTa on task T2A, use:
$> python3 ./data/T2A.train_dev/T2A/public/training_data.txt --checkpoint roberta-base --modelname roberta-finetuned-T2A

The fine-tuned model will be stored in ./roberta-finetuned-T2A/best_checkpoint (note that the best checkpoint is 
decided in terms of macro-F1 in the validation set, and not the validation loss; the rest of the checkpoints are 
removed) 

"""


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return {
        'macro-f1': f1_score(labels, preds, average='macro'),
        'micro-f1': f1_score(labels, preds, average='micro'),
    }


def main(args):
    assert torch.cuda.is_available(), 'cuda is not available'

    datapath = args.trainingfile
    checkpoint = args.checkpoint
    modelout = args.modelname
    if modelout is None:
        modelout = checkpoint+'-finetuned'


    # load the training set, and extract a hold-out validation split of 10% documents (stratified)
    df = pd.read_csv(datapath)
    labels = df['label'].to_frame()
    X_train, X_val = train_test_split(df, stratify=labels, test_size=0.1, random_state=1)
    num_labels = len(pd.unique(labels['label']))
    print('Num classes =', num_labels)

    features = datasets.Features({'label': datasets.Value('int32'), 'text': datasets.Value('string')})
    train = Dataset.from_pandas(df=X_train, split='train', features=features)
    validation = Dataset.from_pandas(df=X_val, split='validation', features=features)

    dataset = DatasetDict({
        'train': train,
        'validation': validation
    })

    # tokenize the dataset
    def tokenize_function(example):
        tokens = tokenizer(example['text'], padding='max_length', truncation=True, max_length=256)
        return tokens

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels).cuda()

    # fine-tuning
    training_args = TrainingArguments(
        modelout,
        learning_rate=1e-5,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=DataCollatorWithPadding(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    clean_checkpoints(modelout, score='eval_macro-f1', higher_is_better=True, verbose=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pre-trained Transformer finetuning')
    parser.add_argument('trainingfile', metavar='PATH', type=str,
                        help='path to the training file')
    parser.add_argument('--checkpoint', metavar='NAME', type=str, default='roberta-base',
                        help="Name of a Huggingface's pre-trained model")
    parser.add_argument('--modelname', metavar='NAME', type=str, default=None,
                        help="The name of the folder where the checkpoints will be dumped. "
                             "Default is None, meaning args.checkpoint+'-finetuned'")
    args = parser.parse_args()

    if not os.path.exists(args.trainingfile):
        raise FileNotFoundError(f'path {args.trainingfile} does not exist')

    main(args)






