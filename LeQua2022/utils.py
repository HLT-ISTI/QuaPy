import os

import numpy as np
import json
from glob import glob
from os.path import join
import shutil


def get_last_checkpoint(path):
    """
    Searches for the last checkpoint and returns the complete path to it

    :param path: a folder containing all the checkpoints generated during the finetuning
    :return: a path to the last checkpoint (the one with the highest step)
    """
    max_step = np.max([int(checkpoint_folder.split('-')[-1]) for checkpoint_folder in glob(join(path, 'checkpoint-*'))])
    return join(path, f'checkpoint-{max_step}')


def choose_best_epoch(json_path, score, higher_is_better):
    with open(json_path, 'rt') as fin:
        js = json.load(fin)

    tr_states = js['log_history']
    tr_states = [state for state in tr_states if 'eval_macro-f1' in state]

    epoch, step, eval_Mf1 = zip(*[(state['epoch'], state['step'], state[score]) for state in tr_states])

    if higher_is_better:
        best_epoch_pos = np.argmax(eval_Mf1)
    else:
        best_epoch_pos = np.argmin(eval_Mf1)

    return int(epoch[best_epoch_pos]), int(step[best_epoch_pos]), eval_Mf1[best_epoch_pos]


def del_checkpoints(path, keep, rename):
    for checkpoint in glob(join(path, 'checkpoint-*')):
        if checkpoint != keep:
            shutil.rmtree(checkpoint)
        else:
            shutil.move(checkpoint, join(path, rename))


def clean_checkpoints(finetuned_path, score='eval_macro-f1', higher_is_better=True, verbose=True):
    def vprint(msg):
        if verbose:
            print(msg)

    vprint(f'Cleaning folder {finetuned_path}')

    checkpoint_path = get_last_checkpoint(finetuned_path)
    vprint(f'> last checkpoint found: {checkpoint_path}')

    json_path = join(checkpoint_path, 'trainer_state.json')
    epoch, step, bestscore = choose_best_epoch(json_path, score=score, higher_is_better=higher_is_better)
    vprint(f'> best epoch (in terms of {score} --{"higher is better" if higher_is_better else "lower is better"}): {epoch}, step={step}, with {score}={bestscore}')

    best_checkpoint = join(finetuned_path, f'checkpoint-{step}')

    vprint(f'> retaining {best_checkpoint} (renamed as best_checkpoint) and cleaning the rest')
    del_checkpoints(finetuned_path, keep=best_checkpoint, rename='best_checkpoint')


