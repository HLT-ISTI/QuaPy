import os
from pathlib import Path
import random

import torch
from torch.nn import MSELoss
from torch.nn.functional import relu

from quapy.method.aggregative import *
from quapy.util import EarlyStop


class QuaNetTrainer(BaseQuantifier):
    """
    Implementation of `QuaNet <https://dl.acm.org/doi/abs/10.1145/3269206.3269287>`_, a neural network for
    quantification. This implementation uses `PyTorch <https://pytorch.org/>`_ and can take advantage of GPU
    for speeding-up the training phase.

    Example:

    >>> import quapy as qp
    >>> from quapy.method.meta import QuaNet
    >>> from quapy.classification.neural import NeuralClassifierTrainer, CNNnet
    >>>
    >>> # use samples of 100 elements
    >>> qp.environ['SAMPLE_SIZE'] = 100
    >>>
    >>> # load the kindle dataset as text, and convert words to numerical indexes
    >>> dataset = qp.datasets.fetch_reviews('kindle', pickle=True)
    >>> qp.data.preprocessing.index(dataset, min_df=5, inplace=True)
    >>>
    >>> # the text classifier is a CNN trained by NeuralClassifierTrainer
    >>> cnn = CNNnet(dataset.vocabulary_size, dataset.n_classes)
    >>> learner = NeuralClassifierTrainer(cnn, device='cuda')
    >>>
    >>> # train QuaNet (QuaNet is an alias to QuaNetTrainer)
    >>> model = QuaNet(learner, qp.environ['SAMPLE_SIZE'], device='cuda')
    >>> model.fit(dataset.training)
    >>> estim_prevalence = model.quantify(dataset.test.instances)

    :param learner: an object implementing `fit` (i.e., that can be trained on labelled data),
        `predict_proba` (i.e., that can generate posterior probabilities of unlabelled examples) and
        `transform` (i.e., that can generate embedded representations of the unlabelled instances).
    :param sample_size: integer, the sample size
    :param n_epochs: integer, maximum number of training epochs
    :param tr_iter_per_poch: integer, number of training iterations before considering an epoch complete
    :param va_iter_per_poch: integer, number of validation iterations to perform after each epoch
    :param lr: float, the learning rate
    :param lstm_hidden_size: integer, hidden dimensionality of the LSTM cells
    :param lstm_nlayers: integer, number of LSTM layers
    :param ff_layers: list of integers, dimensions of the densely-connected FF layers on top of the
        quantification embedding
    :param bidirectional: boolean, indicates whether the LSTM is bidirectional or not
    :param qdrop_p: float, dropout probability
    :param patience: integer, number of epochs showing no improvement in the validation set before stopping the
        training phase (early stopping)
    :param checkpointdir: string, a path where to store models' checkpoints
    :param checkpointname: string (optional), the name of the model's checkpoint
    :param device: string, indicate "cpu" or "cuda"
    """

    def __init__(self,
                 learner,
                 sample_size,
                 n_epochs=100,
                 tr_iter_per_poch=500,
                 va_iter_per_poch=100,
                 lr=1e-3,
                 lstm_hidden_size=64,
                 lstm_nlayers=1,
                 ff_layers=[1024, 512],
                 bidirectional=True,
                 qdrop_p=0.5,
                 patience=10,
                 checkpointdir='../checkpoint',
                 checkpointname=None,
                 device='cuda'):

        assert hasattr(learner, 'transform'), \
            f'the learner {learner.__class__.__name__} does not seem to be able to produce document embeddings ' \
                f'since it does not implement the method "transform"'
        assert hasattr(learner, 'predict_proba'), \
            f'the learner {learner.__class__.__name__} does not seem to be able to produce posterior probabilities ' \
                f'since it does not implement the method "predict_proba"'
        self.learner = learner
        self.sample_size = sample_size
        self.n_epochs = n_epochs
        self.tr_iter = tr_iter_per_poch
        self.va_iter = va_iter_per_poch
        self.lr = lr
        self.quanet_params = {
            'lstm_hidden_size': lstm_hidden_size,
            'lstm_nlayers': lstm_nlayers,
            'ff_layers': ff_layers,
            'bidirectional': bidirectional,
            'qdrop_p': qdrop_p
        }

        self.patience = patience
        if checkpointname is None:
            local_random = random.Random()
            random_code = '-'.join(str(local_random.randint(0, 1000000)) for _ in range(5))
            checkpointname = 'QuaNet-'+random_code
        self.checkpointdir = checkpointdir
        self.checkpoint = os.path.join(checkpointdir, checkpointname)
        self.device = torch.device(device)

        self.__check_params_colision(self.quanet_params, self.learner.get_params())
        self._classes_ = None

    def fit(self, data: LabelledCollection, fit_learner=True):
        """
        Trains QuaNet.

        :param data: the training data on which to train QuaNet. If `fit_learner=True`, the data will be split in
            40/40/20 for training the classifier, training QuaNet, and validating QuaNet, respectively. If
            `fit_learner=False`, the data will be split in 66/34 for training QuaNet and validating it, respectively.
        :param fit_learner: if True, trains the classifier on a split containing 40% of the data
        :return: self
        """
        self._classes_ = data.classes_
        os.makedirs(self.checkpointdir, exist_ok=True)

        if fit_learner:
            classifier_data, unused_data = data.split_stratified(0.4)
            train_data, valid_data = unused_data.split_stratified(0.66)  # 0.66 split of 60% makes 40% and 20%
            self.learner.fit(*classifier_data.Xy)
        else:
            classifier_data = None
            train_data, valid_data = data.split_stratified(0.66)

        # estimate the hard and soft stats tpr and fpr of the classifier
        self.tr_prev = data.prevalence()

        # compute the posterior probabilities of the instances
        valid_posteriors = self.learner.predict_proba(valid_data.instances)
        train_posteriors = self.learner.predict_proba(train_data.instances)

        # turn instances' original representations into embeddings
        valid_data_embed = LabelledCollection(self.learner.transform(valid_data.instances), valid_data.labels, self._classes_)
        train_data_embed = LabelledCollection(self.learner.transform(train_data.instances), train_data.labels, self._classes_)

        self.quantifiers = {
            'cc': CC(self.learner).fit(None, fit_learner=False),
            'acc': ACC(self.learner).fit(None, fit_learner=False, val_split=valid_data),
            'pcc': PCC(self.learner).fit(None, fit_learner=False),
            'pacc': PACC(self.learner).fit(None, fit_learner=False, val_split=valid_data),
        }
        if classifier_data is not None:
            self.quantifiers['emq'] = EMQ(self.learner).fit(classifier_data, fit_learner=False)

        self.status = {
            'tr-loss': -1,
            'va-loss': -1,
            'tr-mae': -1,
            'va-mae': -1,
        }

        nQ = len(self.quantifiers)
        nC = data.n_classes
        self.quanet = QuaNetModule(
            doc_embedding_size=train_data_embed.instances.shape[1],
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
            self._epoch(train_data_embed, train_posteriors, self.tr_iter, epoch_i, early_stop, train=True)
            self._epoch(valid_data_embed, valid_posteriors, self.va_iter, epoch_i, early_stop, train=False)

            early_stop(self.status['va-loss'], epoch_i)
            if early_stop.IMPROVED:
                torch.save(self.quanet.state_dict(), checkpoint)
            elif early_stop.STOP:
                print(f'training ended by patience exhausted; loading best model parameters in {checkpoint} '
                      f'for epoch {early_stop.best_epoch}')
                self.quanet.load_state_dict(torch.load(checkpoint))
                break

        return self

    def _get_aggregative_estims(self, posteriors):
        label_predictions = np.argmax(posteriors, axis=-1)
        prevs_estim = []
        for quantifier in self.quantifiers.values():
            predictions = posteriors if quantifier.probabilistic else label_predictions
            prevs_estim.extend(quantifier.aggregate(predictions))

        # there is no real need for adding static estims like the TPR or FPR from training since those are constant

        return prevs_estim

    def quantify(self, instances):
        posteriors = self.learner.predict_proba(instances)
        embeddings = self.learner.transform(instances)
        quant_estims = self._get_aggregative_estims(posteriors)
        self.quanet.eval()
        with torch.no_grad():
            prevalence = self.quanet.forward(embeddings, posteriors, quant_estims)
            if self.device == torch.device('cuda'):
                prevalence = prevalence.cpu()
            prevalence = prevalence.numpy().flatten()
        return prevalence

    def _epoch(self, data: LabelledCollection, posteriors, iterations, epoch, early_stop, train):
        mse_loss = MSELoss()

        self.quanet.train(mode=train)
        losses = []
        mae_errors = []
        if train==False:
            prevpoints = F.get_nprevpoints_approximation(iterations, self.quanet.n_classes)
            iterations = F.num_prevalence_combinations(prevpoints, self.quanet.n_classes)
            with qp.util.temp_seed(0):
                sampling_index_gen = data.artificial_sampling_index_generator(self.sample_size, prevpoints)
        else:
            sampling_index_gen = [data.sampling_index(self.sample_size, *prev) for prev in
                                  F.uniform_simplex_sampling(data.n_classes, iterations)]
        pbar = tqdm(sampling_index_gen, total=iterations) if train else sampling_index_gen

        for it, index in enumerate(pbar):
            sample_data = data.sampling_from_index(index)
            sample_posteriors = posteriors[index]
            quant_estims = self._get_aggregative_estims(sample_posteriors)
            ptrue = torch.as_tensor([sample_data.prevalence()], dtype=torch.float, device=self.device)
            if train:
                self.optim.zero_grad()
                phat = self.quanet.forward(sample_data.instances, sample_posteriors, quant_estims)
                loss = mse_loss(phat, ptrue)
                mae = mae_loss(phat, ptrue)
                loss.backward()
                self.optim.step()
            else:
                with torch.no_grad():
                    phat = self.quanet.forward(sample_data.instances, sample_posteriors, quant_estims)
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
                                     f'epoch={epoch} [it={it}/{iterations}]\t'
                                     f'tr-mseloss={self.status["tr-loss"]:.5f} tr-maeloss={self.status["tr-mae"]:.5f}\t'
                                     f'val-mseloss={self.status["va-loss"]:.5f} val-maeloss={self.status["va-mae"]:.5f} '
                                     f'patience={early_stop.patience}/{early_stop.PATIENCE_LIMIT}')

    def get_params(self, deep=True):
        return {**self.learner.get_params(), **self.quanet_params}

    def set_params(self, **parameters):
        learner_params = {}
        for key, val in parameters.items():
            if key in self.quanet_params:
                self.quanet_params[key] = val
            else:
                learner_params[key] = val
        self.learner.set_params(**learner_params)

    def __check_params_colision(self, quanet_params, learner_params):
        quanet_keys = set(quanet_params.keys())
        learner_keys = set(learner_params.keys())
        intersection = quanet_keys.intersection(learner_keys)
        if len(intersection) > 0:
            raise ValueError(f'the use of parameters {intersection} is ambiguous sine those can refer to '
                             f'the parameters of QuaNet or the learner {self.learner.__class__.__name__}')

    def clean_checkpoint(self):
        """
        Removes the checkpoint
        """
        os.remove(self.checkpoint)

    def clean_checkpoint_dir(self):
        """
        Removes anything contained in the checkpoint directory
        """
        import shutil
        shutil.rmtree(self.checkpointdir, ignore_errors=True)

    @property
    def classes_(self):
        return self._classes_


def mae_loss(output, target):
    """
    Torch-like wrapper for the Mean Absolute Error

    :param output: predictions
    :param target: ground truth values
    :return: mean absolute error loss
    """
    return torch.mean(torch.abs(output - target))


class QuaNetModule(torch.nn.Module):
    """
    Implements the `QuaNet <https://dl.acm.org/doi/abs/10.1145/3269206.3269287>`_ forward pass.
    See :class:`QuaNetTrainer` for training QuaNet.

    :param doc_embedding_size: integer, the dimensionality of the document embeddings
    :param n_classes: integer, number of classes
    :param stats_size: integer, number of statistics estimated by simple quantification methods
    :param lstm_hidden_size: integer, hidden dimensionality of the LSTM cell
    :param lstm_nlayers: integer, number of LSTM layers
    :param ff_layers: list of integers, dimensions of the densely-connected FF layers on top of the
        quantification embedding
    :param bidirectional: boolean, whether or not to use bidirectional LSTM
    :param qdrop_p: float, dropout probability
    :param order_by: integer, class for which the document embeddings are to be sorted
    """

    def __init__(self,
                 doc_embedding_size,
                 n_classes,
                 stats_size,
                 lstm_hidden_size=64,
                 lstm_nlayers=1,
                 ff_layers=[1024, 512],
                 bidirectional=True,
                 qdrop_p=0.5,
                 order_by=0):

        super().__init__()

        self.n_classes = n_classes
        self.order_by = order_by
        self.hidden_size = lstm_hidden_size
        self.nlayers = lstm_nlayers
        self.bidirectional = bidirectional
        self.ndirections = 2 if self.bidirectional else 1
        self.qdrop_p = qdrop_p
        self.lstm = torch.nn.LSTM(doc_embedding_size + n_classes,  # +n_classes stands for the posterior probs. (concatenated)
                                  lstm_hidden_size, lstm_nlayers, bidirectional=bidirectional,
                                  dropout=qdrop_p, batch_first=True)
        self.dropout = torch.nn.Dropout(self.qdrop_p)

        lstm_output_size = self.hidden_size * self.ndirections
        ff_input_size = lstm_output_size + stats_size
        prev_size = ff_input_size
        self.ff_layers = torch.nn.ModuleList()
        for lin_size in ff_layers:
            self.ff_layers.append(torch.nn.Linear(prev_size, lin_size))
            prev_size = lin_size
        self.output = torch.nn.Linear(prev_size, n_classes)

    @property
    def device(self):
        return torch.device('cuda') if next(self.parameters()).is_cuda else torch.device('cpu')

    def _init_hidden(self):
        directions = 2 if self.bidirectional else 1
        var_hidden = torch.zeros(self.nlayers * directions, 1, self.hidden_size)
        var_cell = torch.zeros(self.nlayers * directions, 1, self.hidden_size)
        if next(self.lstm.parameters()).is_cuda:
            var_hidden, var_cell = var_hidden.cuda(), var_cell.cuda()
        return var_hidden, var_cell

    def forward(self, doc_embeddings, doc_posteriors, statistics):
        device = self.device
        doc_embeddings = torch.as_tensor(doc_embeddings, dtype=torch.float, device=device)
        doc_posteriors = torch.as_tensor(doc_posteriors, dtype=torch.float, device=device)
        statistics = torch.as_tensor(statistics, dtype=torch.float, device=device)

        if self.order_by is not None:
            order = torch.argsort(doc_posteriors[:, self.order_by])
            doc_embeddings = doc_embeddings[order]
            doc_posteriors = doc_posteriors[order]

        embeded_posteriors = torch.cat((doc_embeddings, doc_posteriors), dim=-1)

        # the entire set represents only one instance in quapy contexts, and so the batch_size=1
        # the shape should be (1, number-of-instances, embedding-size + n_classes)
        embeded_posteriors = embeded_posteriors.unsqueeze(0)

        self.lstm.flatten_parameters()
        _, (rnn_hidden,_) = self.lstm(embeded_posteriors, self._init_hidden())
        rnn_hidden = rnn_hidden.view(self.nlayers, self.ndirections, 1, self.hidden_size)
        quant_embedding = rnn_hidden[0].view(-1)
        quant_embedding = torch.cat((quant_embedding, statistics))

        abstracted = quant_embedding.unsqueeze(0)
        for linear in self.ff_layers:
            abstracted = self.dropout(relu(linear(abstracted)))

        logits = self.output(abstracted).view(1, -1)
        prevalence = torch.softmax(logits, -1)

        return prevalence





