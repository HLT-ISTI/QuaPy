import argparse
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from os.path import join
import os
import shutil
from tqdm import tqdm
from glob import glob
from pathlib import Path
import re
import datasets
from datasets import Dataset
import pandas as pd


"""
This scripts takes a pre-trained model (a fine-tuned one) and generates numerical representations for all
samples in the dataset. The representations are saved in npy-txt plain format.
"""


def tokenize_function(tokenizer, example):
    tokens = tokenizer(example['text'], padding='max_length', truncation=True, max_length=256, return_tensors='pt')
    return {
        'input_ids': tokens.input_ids.cuda(),
        'attention_mask': tokens.attention_mask.cuda()
    }


def save_samples_as_txt(vectors, labels, path):
    cols = list(np.arange(vectors.shape[1]))
    if labels is not None:
        labels = labels.values
        cols = ['label']+cols
        vectors = np.hstack([labels, vectors])
    df = pd.DataFrame(vectors, columns=cols)
    if labels is not None:
        df["label"] = pd.to_numeric(df["label"], downcast='integer')
    df.to_csv(path, index=False)


def load_single_sample(path):
    df = pd.read_csv(path)
    labelled = 'label' in df.columns
    labels = df.pop('label').to_frame() if labelled else None
    features = datasets.Features({'text': datasets.Value('string')})
    sample = Dataset.from_pandas(df=df, features=features)
    return sample, labels


def transform_sample(model, tokenizer, instances, batch_size=50):
    ndocs = len(instances)
    batches = ndocs // batch_size
    assert ndocs % batches == 0, 'fragmented last bach not supported'

    transformations = []
    for batch_id in range(0, ndocs, batch_size):

        batch_instances = instances[batch_id:batch_id + batch_size]
        tokenized_dataset = tokenize_function(tokenizer, batch_instances)
        out = model(**tokenized_dataset, output_hidden_states=True)

        hidden_states = torch.stack(out.hidden_states)
        # we average all embedding representations for the special [CLS] token
        # the hidden_states tensor has shape:
        #   [num_layers (13), batch_size (50), sequence_length (256), embedding-dimension (768)]
        # we get rid of the first layer (see "[1:" in the indexing), since the (non-contextualized)
        # ...first representation of [CLS] is the same irrespective of the layer
        # the [CLS] token lies in the first position (see the "0" in the indexing) in the sequence
        all_layer_cls = hidden_states[1:, :, 0, :]
        average_cls = torch.mean(all_layer_cls, dim=0)
        transformations.append(average_cls.cpu().numpy())

    transformations = np.vstack(transformations)
    return transformations


def transform_folder(inputfolder, outputfolder, model, tokenizer, skip_if_exist=True):
    os.makedirs(outputfolder, exist_ok=True)
    files = list(glob(join(inputfolder, '**'), recursive=True))
    pbar = tqdm(files, total=len(files))
    for inpath in pbar:
        outpath = inpath.replace(inputfolder, outputfolder)
        os.makedirs(Path(outpath).parent, exist_ok=True)
        if skip_if_exist and os.path.exists(outpath):
            pbar.set_description(f'skipping already existing file: {outpath}')
            continue
        filename = Path(outpath).name
        if filename.endswith('.txt'):
            if filename == 'training_data.txt' or re.match(r"^\d+.txt$", filename):
                pbar.set_description(f'processing >> {inpath}')
                instances, labels = load_single_sample(inpath)
                transformed = transform_sample(model, tokenizer, instances)
                save_samples_as_txt(transformed, labels, outpath)
            else:
                pbar.set_description(f'copying {inpath} --> {outpath}')
                shutil.copyfile(inpath, outpath)


def main(args):
    assert torch.cuda.is_available(), 'cuda is not available'

    checkpoint = args.checkpoint

    num_labels = 2 if args.task == 'T2A' else 28

    with torch.no_grad():
        print('loading', checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels).cuda()

        transform_folder(args.inputfolder, args.outputfolder, model, tokenizer)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract dense features for all samples using a pre-trained model')
    parser.add_argument('checkpoint', metavar='NAME', type=str,
                        help="Path to a pre-trained fine-tuned model, or name of a Huggingface's pre-trained model")
    parser.add_argument('inputfolder', metavar='PATH', type=str,
                        help="Path to the folder containing the data samples (assuming LeQua's file structure)")
    parser.add_argument('outputfolder', metavar='PATH', type=str,
                        help="Path to the folder where to dump the representations. "
                             "This folder will mimic the original LeQua's file structure.")
    parser.add_argument('task', metavar='TASK', type=str, choices=['T2A', 'T2B'],
                        help='The name of the raw-document task (T2A, T2B)')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Pretrained model {args.checkpoint} not found')
    if not os.path.exists(args.inputfolder):
        raise FileNotFoundError(f'folder {args.inputfolder} does not exist')

    main(args)





