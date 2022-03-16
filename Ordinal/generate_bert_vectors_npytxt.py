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

from Ordinal.utils import load_samples_folder, load_single_sample_as_csv



def tokenize_function(example):
    tokens = tokenizer(example['review'], padding='max_length', truncation=True, max_length=64 if debug else None, return_tensors='pt')
    return {
        'input_ids': tokens.input_ids.cuda(),
        'attention_mask': tokens.attention_mask.cuda()
    }


def save_samples_as_txt(tensors, labels, path):
    vectors = tensors
    labels = labels.values
    vec_lab = np.hstack([labels, vectors])
    n_cols = vectors.shape[1]
    np.savetxt(path, vec_lab, fmt=['%d']+['%f']*n_cols)


def transform_sample(instances, labels, outpath, batch_size=50):
    ndocs = len(labels)
    batches = ndocs // batch_size
    assert ndocs % batches == 0, 'fragmented last bach not supported'

    transformations = []
    for batch_id in range(0, ndocs, batch_size):

        batch_instances = instances[batch_id:batch_id + batch_size]

        tokenized_dataset = tokenize_function(batch_instances)
        out = model(**tokenized_dataset, output_hidden_states=True)

        if generation_mode == 'posteriors':
            logits = out.logits
            posteriors = torch.softmax(logits, dim=-1)
            transformed = posteriors
        elif generation_mode == 'last':
            hidden_states = out.hidden_states
            last_layer_cls = hidden_states[-1][:, 0, :]
            transformed = last_layer_cls
        elif generation_mode == 'average':
            hidden_states = out.hidden_states
            hidden_states = torch.stack(hidden_states)
            all_layer_cls = hidden_states[:, :, 0, :]
            average_cls = torch.mean(all_layer_cls, dim=0)
            transformed = average_cls
        else:
            raise NotImplementedError()

        transformations.append(transformed.cpu().numpy())

    transformations = np.vstack(transformations)
    save_samples_as_txt(transformations, labels, outpath)


def transform_folder_samples(protocol, splitname):
    in_folder = join(datapath, domain, protocol, splitname)
    out_folder = join(datapath, outname, protocol, splitname)
    total = 1000 if splitname.startswith('dev') else 5000

    for i, (instances, labels) in tqdm(enumerate(
            load_samples_folder(in_folder, load_fn=load_single_sample_as_csv)), desc=f'{protocol} {splitname}', total=total):
        transform_sample(instances, labels, outpath=join(out_folder, f'{i}.txt'))


def get_best_checkpoint(checkpointdir):
    from glob import glob
    steps = []
    for folder in glob(f'{checkpointdir}/checkpoint-*'):
        step=int(folder.split('checkpoint-')[1])
        steps.append(step)
    assert len(steps) <= 2, 'unexpected number of steps, only two where expected (the best one and the last one)'
    choosen = f'{checkpointdir}/checkpoint-{min(steps)}'
    return choosen


if __name__ == '__main__':
    debug = False
    assert torch.cuda.is_available(), 'cuda is not available'

    checkpoint='roberta-base-finetuned'
    generation_mode = 'posteriors'

    # n_args = len(sys.argv)
    # assert n_args==3, 'wrong arguments, expected: <checkpoint> <generation-mode>\n' \
    #                   '\tgeneration-mode: last (last layer), ave (average pooling), or posteriors (posterior probabilities)'

    # checkpoint = sys.argv[1]  #e.g., 'bert-base-uncased'
    # generation_mode = sys.argv[2]  # e.g., 'last'
    
    assert 'finetuned' in checkpoint, 'looks like this model is not finetuned'

    checkpoint = get_best_checkpoint(checkpoint)

    num_labels = 5

    datapath = './data'
    domain = 'Books'
    protocols = ['app']  # ['app', 'npp']

    assert generation_mode in ['last', 'average', 'posteriors'], 'unknown generation_model'
    outname = domain + f'-{checkpoint}-{generation_mode}'

    with torch.no_grad():
        print('loading', checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels).cuda()

        os.makedirs(join(datapath, outname), exist_ok=True)

        print('transforming the training set')
        instances, labels = load_single_sample_as_csv(join(datapath, domain), 'training_data')
        transform_sample(instances, labels, join(datapath, outname, 'training_data.txt'))
        print('[done]')

        for protocol in protocols:
            in_path = join(datapath, domain, protocol)
            out_path = join(datapath, outname, protocol)
            os.makedirs(out_path, exist_ok=True)
            os.makedirs(join(out_path, 'dev_samples'), exist_ok=True)
            os.makedirs(join(out_path, 'test_samples'), exist_ok=True)
            shutil.copyfile(join(in_path, 'dev_prevalences.txt'), join(out_path, 'dev_prevalences.txt'))
            shutil.copyfile(join(in_path, 'test_prevalences.txt'), join(out_path, 'test_prevalences.txt'))

            print('processing', protocol)
            transform_folder_samples(protocol, 'dev_samples')
            transform_folder_samples(protocol, 'test_samples')






