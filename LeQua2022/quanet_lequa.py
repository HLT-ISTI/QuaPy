import os
import random
import torch
from torch.nn import MSELoss

from data import ResultSubmission, load_npy_documents
from quapy.method.neural import QuaNetModule, QuaNetTrainer, mae_loss
from quapy.method.aggregative import *
from quapy.util import EarlyStop
from sklearn.base import clone


def _gen_load_samples_with_groudtruth(prop_from, prop_to, path_dir:str, ground_truth:ResultSubmission, ext, load_fn):
    n_samples = len(ground_truth)
    id_from, id_to = int(n_samples*prop_from), int(n_samples*prop_to)
    for id, prevalence in ground_truth.iterrows():
        if id_from <= id < id_to:
            sample, _ = load_fn(os.path.join(path_dir, f'{id}.{ext}'))
            yield sample, prevalence


class QuaNetTrainerLeQua2022(QuaNetTrainer):

    def __init__(self,
                 learner,
                 sample_size,
                 val_samples_dir,
                 groundtruth_path,
                 n_epochs=500,
                 lr=1e-3,
                 lstm_hidden_size=64,
                 lstm_nlayers=1,
                 ff_layers=[1024, 512],
                 bidirectional=True,
                 qdrop_p=0.5,
                 patience=10,
                 checkpointdir='../checkpoint_quanet',
                 checkpointname=None,
                 device='cuda'):

        assert hasattr(learner, 'predict_proba'), \
            f'the learner {learner.__class__.__name__} does not seem to be able to produce posterior probabilities ' \
                f'since it does not implement the method "predict_proba"'
        self.learner = learner
        self.sample_size = sample_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.quanet_params = {
            'lstm_hidden_size': lstm_hidden_size,
            'lstm_nlayers': lstm_nlayers,
            'ff_layers': ff_layers,
            'bidirectional': bidirectional,
            'qdrop_p': qdrop_p
        }
        self.val_samples_dir = val_samples_dir
        self.ground_truth = ResultSubmission.load(groundtruth_path)

        self.patience = patience
        if checkpointname is None:
            local_random = random.Random()
            random_code = '-'.join(str(local_random.randint(0, 1000000)) for _ in range(5))
            checkpointname = 'QuaNet-'+random_code
        self.checkpointdir = checkpointdir
        self.checkpoint = os.path.join(checkpointdir, checkpointname)
        self.device = torch.device(device)

        self._check_params_colision(self.quanet_params, self.learner.get_params())
        self._classes_ = None

    def train_sample_generator(self):
        return _gen_load_samples_with_groudtruth(
            prop_from=0.0,
            prop_to=0.9,
            path_dir=self.val_samples_dir,
            ground_truth=self.ground_truth,
            ext='npy',
            load_fn=load_npy_documents
        )

    def val_sample_generator(self):
        return _gen_load_samples_with_groudtruth(
            prop_from=0.9,
            prop_to=1.0,
            path_dir=self.val_samples_dir,
            ground_truth=self.ground_truth,
            ext='npy',
            load_fn=load_npy_documents
        )

    def fit(self, data: LabelledCollection, fit_learner=True):
        assert fit_learner, 'fit_learner has to be True'
        self._classes_ = data.classes_
        os.makedirs(self.checkpointdir, exist_ok=True)

        self.learner.fit(*data.Xy)

        self.quantifiers = {
            'cc': CC(self.learner).fit(data, fit_learner=False),
            'acc': ACC(clone(self.learner)).fit(data),
            'pcc': PCC(self.learner).fit(data, fit_learner=False),
            'pacc': PACC(clone(self.learner)).fit(data),
            'emq': EMQ(self.learner).fit(data, fit_learner=False)
        }

        self.status = {
            'tr-loss': -1,
            'va-loss': -1,
            'tr-mae': -1,
            'va-mae': -1,
        }

        nQ = len(self.quantifiers)
        nC = data.n_classes
        self.quanet = QuaNetModule(
            doc_embedding_size=data.instances.shape[1],
            n_classes=data.n_classes,
            stats_size=nQ*nC,
            order_by=0 if data.binary else None,
            **self.quanet_params
        ).to(self.device)
        print(self.quanet)

        self.optim = torch.optim.Adam(self.quanet.parameters(), lr=self.lr)
        early_stop = EarlyStop(self.patience, lower_is_better=True)

        checkpoint = self.checkpoint

        for epoch_i in range(1, self.n_epochs):
            self._epoch(epoch_i, early_stop, train=True)
            self._epoch(epoch_i, early_stop, train=False)

            early_stop(self.status['va-loss'], epoch_i)
            if early_stop.IMPROVED:
                torch.save(self.quanet.state_dict(), checkpoint)
            elif early_stop.STOP:
                print(f'training ended by patience exhausted; loading best model parameters in {checkpoint} '
                      f'for epoch {early_stop.best_epoch}')
                self.quanet.load_state_dict(torch.load(checkpoint))
                break

        return self

    def quantify(self, instances):
        posteriors = self.learner.predict_proba(instances)
        quant_estims = self._get_aggregative_estims(posteriors)
        self.quanet.eval()
        with torch.no_grad():
            prevalence = self.quanet.forward(instances, posteriors, quant_estims)
            if self.device == torch.device('cuda'):
                prevalence = prevalence.cpu()
            prevalence = prevalence.numpy().flatten()
        return prevalence

    def _epoch(self, epoch, early_stop, train):
        mse_loss = MSELoss()

        sample_generator = self.train_sample_generator if train else self.val_sample_generator

        self.quanet.train(mode=train)
        losses = []
        mae_errors = []
        pbar = tqdm(sample_generator())

        for it, (sample_data, sample_prev) in enumerate(pbar):
            sample_posteriors = self.learner.predict_proba(sample_data)
            quant_estims = self._get_aggregative_estims(sample_posteriors)
            ptrue = torch.as_tensor([sample_prev], dtype=torch.float, device=self.device)
            if train:
                self.optim.zero_grad()
                phat = self.quanet.forward(sample_data, sample_posteriors, quant_estims)
                loss = mse_loss(phat, ptrue)
                mae = mae_loss(phat, ptrue)
                loss.backward()
                self.optim.step()
            else:
                with torch.no_grad():
                    phat = self.quanet.forward(sample_data, sample_posteriors, quant_estims)
                    loss = mse_loss(phat, ptrue)
                    mae = mae_loss(phat, ptrue)

            losses.append(loss.item())
            mae_errors.append(mae.item())

            mse = np.mean(losses)
            mae = np.mean(mae_errors)
            if train:
                self.status['tr-loss'] = mse
                self.status['tr-mae'] = mae
            else:
                self.status['va-loss'] = mse
                self.status['va-mae'] = mae

            if train:
                pbar.set_description(f'[QuaNet] '
                                     f'epoch={epoch} [it={it}] '
                                     f'tr-mseloss={self.status["tr-loss"]:.5f} tr-maeloss={self.status["tr-mae"]:.5f} '
                                     f'val-mseloss={self.status["va-loss"]:.5f} val-maeloss={self.status["va-mae"]:.5f} '
                                     f'patience={early_stop.patience}/{early_stop.PATIENCE_LIMIT}')