import argparse
import quapy as qp
from data import ResultSubmission, evaluate_submission
import constants
import os

"""
LeQua2022 Official evaluation script 
"""

def main(args):
    if args.task in {'T1A', 'T2A'}:
        qp.environ['SAMPLE_SIZE'] = constants.TXA_SAMPLE_SIZE
    if args.task in {'T1B', 'T2B'}:
        qp.environ['SAMPLE_SIZE'] = constants.TXB_SAMPLE_SIZE
    true_prev = ResultSubmission.load(args.true_prevalences)
    pred_prev = ResultSubmission.load(args.pred_prevalences)
    mae, mrae = evaluate_submission(true_prev, pred_prev)
    print(f'MAE: {mae:.4f}')
    print(f'MRAE: {mrae:.4f}')

    if args.output is not None:
        outdir = os.path.dirname(args.output)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        with open(args.output, 'wt') as foo:
            foo.write(f'MAE: {mae:.4f}\n')
            foo.write(f'MRAE: {mrae:.4f}\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='LeQua2022 official evaluation script')
    parser.add_argument('task', metavar='TASK', type=str, choices=['T1A', 'T1B', 'T2A', 'T2B'],
                        help='Task name (T1A, T1B, T2A, T2B)')
    parser.add_argument('true_prevalences', metavar='TRUE-PREV-PATH', type=str,
                        help='Path of ground truth prevalence values file (.csv)')
    parser.add_argument('pred_prevalences', metavar='PRED-PREV-PATH', type=str,
                        help='Path of predicted prevalence values file (.csv)')
    parser.add_argument('--output', metavar='SCORES-PATH', type=str, default=None,
                        help='Path where to store the evaluation scores')
    args = parser.parse_args()

    main(args)
