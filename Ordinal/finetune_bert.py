import sys
import numpy as np
import datasets
import torch.cuda
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding, BertTokenizer
from datasets import list_datasets, list_metrics, load_dataset, Dataset, DatasetDict, load_metric
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import pandas as pd
import csv


def tokenize_function(example):
    tokens = tokenizer(example['review'], padding='max_length', truncation=True, max_length=64 if debug else 256)
    return tokens


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return {
        'macro-f1': f1_score(labels, preds, average='macro'),
        'micro-f1': f1_score(labels, preds, average='micro'),
    }


if __name__ == '__main__':
    debug = False
    assert torch.cuda.is_available(), 'cuda is not available'

    n_args = len(sys.argv)
    assert n_args==3, 'wrong arguments, expected: <training-path> <transformer-name>'

    datapath = sys.argv[1]  # './data/Books/training_data.txt'
    checkpoint = sys.argv[2]  #e.g., 'bert-base-uncased' or 'distilbert-base-uncased' or 'roberta-base'
    modelout = checkpoint+'-finetuned'

    # load the training set, and extract a held-out validation split of 1000 documents (stratified)
    df = pd.read_csv(datapath, sep='\t', names=['labels', 'review'], quoting=csv.QUOTE_NONE)
    labels = df['labels'].to_frame()
    X_train, X_val = train_test_split(df, stratify=labels, test_size=1000, random_state=1)
    num_labels = len(pd.unique(labels['labels']))

    features = datasets.Features({'labels': datasets.Value('int32'), 'review': datasets.Value('string')})
    train = Dataset.from_pandas(df=X_train, split='train', features=features)
    validation = Dataset.from_pandas(df=X_val, split='validation', features=features)

    dataset = DatasetDict({
        'train': train.select(range(500)) if debug else train,
        'validation': validation.select(range(500)) if debug else validation
    })

    # tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    print(tokenized_datasets)
    print(tokenized_datasets['train'][0]['labels'])
    print(tokenized_datasets['train'][0]['review'])
    print(tokenized_datasets['train'][0]['input_ids'])
    print(len(tokenized_datasets['train'][0]['input_ids']))
    # print(tokenized_datasets['train'][0]['token_type_ids'])
    # print(tokenized_datasets['train'][0]['attention_mask'])

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels).cuda()

    # fine-tuning
    training_args = TrainingArguments(
        modelout,
        learning_rate=2e-5,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        # eval_steps=10,
        save_total_limit=1,
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






