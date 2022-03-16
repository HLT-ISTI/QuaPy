import numpy as np
from Ordinal.evaluation import smoothness
from Ordinal.utils import load_samples_folder, load_single_sample_pkl

from os.path import join
from tqdm import tqdm


def partition_by_smoothness(split):
    assert split in ['dev', 'test'], 'invalid split name'
    total=1000 if split=='dev' else 5000
    smooths = []
    folderpath = join(datapath, domain, 'app', f'{split}_samples')
    for sample in tqdm(load_samples_folder(folderpath, load_fn=load_single_sample_pkl), total=total):
        smooths.append(smoothness(sample.prevalence()))
    smooths = np.asarray(smooths)
    order = np.argsort(smooths)
    nD = len(order)
    low2high_smooth = np.array_split(order, 5) 
    all_drift = np.arange(nD)
    for i, smooth_idx in enumerate(low2high_smooth):
        block = smooths[smooth_idx]
        print(f'smooth block {i}: shape={smooth_idx.shape}, interval=[{block.min()}, {block.max()}] mean={block.mean()}')
        np.save(join(datapath, domain, 'app', f'smooth{i}.{split}.id.npy'), smooth_idx)
    np.save(join(datapath, domain, 'app', f'all.{split}.id.npy'), all_drift)


#domain = 'Books-tfidf'
domain = 'Books-roberta-base-finetuned-pkl/checkpoint-1188-average'
datapath = './data'

#training = pickle.load(open(join(datapath,domain,'training_data.pkl'), 'rb'))

partition_by_smoothness('dev')
partition_by_smoothness('test')

