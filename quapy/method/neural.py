import os
from pathlib import Path
import torch
from torch.nn import MSELoss
from torch.nn.functional import relu
from tqdm import tqdm
from method.aggregative import *
from util import EarlyStop


class QuaNetTrainer(BaseQuantifier):

    def __init__(self,
                 learner,
                 sample_size,
                 n_epochs=500,
                 tr_iter_per_poch=200,
                 va_iter_per_poch=21,
                 lr=1e-3,
                 lstm_hidden_size=64,
                 lstm_nlayers=1,
                 ff_layers=[1024, 512],
                 bidirectional=True,
                 qdrop_p=0.5,
                 patience=10, checkpointpath='../checkpoint/quanet.dat', device='cuda'):
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
        self.checkpointpath = checkpointpath
        os.makedirs(Path(checkpointpath).parent, exist_ok=True)
        self.device = torch.device(device)

        self.__check_params_colision(self.quanet_params, self.learner.get_params())

    def fit(self, data: LabelledCollection, fit_learner=True, *args):
        """
        :param data: the training data on which to train QuaNet. If fit_learner=True, the data will be split in
        40/40/20 for training the classifier, training QuaNet, and validating QuaNet, respectively. If
        fit_learner=False, the data will be split in 66/34 for training QuaNet and validating it, respectively.
        :param fit_learner: if true, trains the classifier on a split containing 40% of the data
        :param args: unused
        :return: self
        """
        # split: 40% for training classification, 40% for training quapy, and 20% for validating quapy
        self.learner, unused_data = \
            training_helper(self.learner, data, fit_learner,  ensure_probabilistic=True, val_split=0.6)
        train_data, valid_data = unused_data.split_stratified(0.66)  # 0.66 split of 60% makes 40% and 20%

        # compute the posterior probabilities of the instances
        valid_posteriors = self.learner.predict_proba(valid_data.instances)
        train_posteriors = self.learner.predict_proba(train_data.instances)

        # turn instances' indexes into embeddings
        valid_data.instances = self.learner.transform(valid_data.instances)
        train_data.instances = self.learner.transform(train_data.instances)

        # estimate the hard and soft stats tpr and fpr of the classifier
        self.tr_prev = data.prevalence()

        self.quantifiers = {
            'cc': CC(self.learner).fit(data, fit_learner=False),
            'acc': ACC(self.learner).fit(data, fit_learner=False),
            'pcc': PCC(self.learner).fit(data, fit_learner=False),
            'pacc': PACC(self.learner).fit(data, fit_learner=False),
            'emq': EMQ(self.learner).fit(data, fit_learner=False),
        }

        self.status = {
            'tr-loss': -1,
            'va-loss': -1,
        }

        nQ = len(self.quantifiers)
        nC = data.n_classes
        self.quanet = QuaNetModule(
            doc_embedding_size=train_data.instances.shape[1],
            n_classes=data.n_classes,
            stats_size=nQ*nC + 2*nC*nC,
            order_by=0 if data.binary else None,
            **self.quanet_params
        ).to(self.device)

        self.optim = torch.optim.Adam(self.quanet.parameters(), lr=self.lr)
        early_stop = EarlyStop(self.patience, lower_is_better=True)

        checkpoint = self.checkpointpath

        for epoch_i in range(1, self.n_epochs):
            self.epoch(train_data, train_posteriors, self.tr_iter, epoch_i, early_stop, train=True)
            self.epoch(valid_data, valid_posteriors, self.va_iter, epoch_i, early_stop, train=False)

            early_stop(self.status['va-loss'], epoch_i)
            if early_stop.IMPROVED:
                torch.save(self.quanet.state_dict(), checkpoint)
            elif early_stop.STOP:
                print(f'training ended by patience exhausted; loading best model parameters in {checkpoint} '
                      f'for epoch {early_stop.best_epoch}')
                self.quanet.load_state_dict(torch.load(checkpoint))
                self.epoch(valid_data, valid_posteriors, self.va_iter, epoch_i, early_stop, train=True)
                break

        return self

    def get_aggregative_estims(self, posteriors):
        label_predictions = np.argmax(posteriors, axis=-1)
        prevs_estim = []
        for quantifier in self.quantifiers.values():
            predictions = posteriors if isprobabilistic(quantifier) else label_predictions
            prevs_estim.extend(quantifier.aggregate(predictions))

        # add the class-conditional predictions P(y'i|yj) from ACC and PACC
        prevs_estim.extend(self.quantifiers['acc'].Pte_cond_estim_.flatten())
        prevs_estim.extend(self.quantifiers['pacc'].Pte_cond_estim_.flatten())

        return prevs_estim

    def quantify(self, instances, *args):
        posteriors = self.learner.predict_proba(instances)
        embeddings = self.learner.transform(instances)
        quant_estims = self.get_aggregative_estims(posteriors)
        self.quanet.eval()
        with torch.no_grad():
            prevalence = self.quanet.forward(embeddings, posteriors, quant_estims).item()
        return prevalence

    def epoch(self, data: LabelledCollection, posteriors, iterations, epoch, early_stop, train):
        mse_loss = MSELoss()
        prevpoints = F.get_nprevpoints_approximation(iterations, self.quanet.n_classes)

        self.quanet.train(mode=train)
        losses = []
        pbar = tqdm(data.artificial_sampling_index_generator(self.sample_size, prevpoints))
        for it, index in enumerate(pbar):
            sample_data = data.sampling_from_index(index)
            sample_posteriors = posteriors[index]
            quant_estims = self.get_aggregative_estims(sample_posteriors)
            ptrue = torch.as_tensor([sample_data.prevalence()], dtype=torch.float, device=self.device)
            if train:
                self.optim.zero_grad()
                phat = self.quanet.forward(sample_data.instances, sample_posteriors, quant_estims)
                loss = mse_loss(phat, ptrue)
                loss.backward()
                self.optim.step()
            else:
                with torch.no_grad():
                    phat = self.quanet.forward(sample_data.instances, sample_posteriors, quant_estims)
                    loss = mse_loss(phat, ptrue)

            losses.append(loss.item())

            self.status['tr-loss' if train else 'va-loss'] = np.mean(losses[-10:])
            pbar.set_description(f'[QuaNet][{"training" if train else "validating"}] '
                                 f'epoch={epoch} [it={it}/{iterations}]\t'
                                 f'tr-loss={self.status["tr-loss"]:.5f} '
                                 f'val-loss={self.status["va-loss"]:.5f} '
                                 f'patience={early_stop.patience}/{early_stop.PATIENCE_LIMIT}')

    def get_params(self, deep=True):
        return {**self.learner.get_params(), **self.quanet_params}

    def set_params(self, **parameters):
        learner_params={}
        for key, val in parameters:
            if key in self.quanet_params:
                self.quanet_params[key]=val
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


class QuaNetModule(torch.nn.Module):
    def __init__(self,
                 doc_embedding_size,
                 n_classes,
                 stats_size,
                 lstm_hidden_size=64,
                 lstm_nlayers=1,
                 ff_layers=[1024, 512],
                 bidirectional=True,
                 qdrop_p=0.5,
                 order_by=None):
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

    def init_hidden(self):
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
        # the shape should be (1, number-of-instances, embedding-size + 1)
        embeded_posteriors = embeded_posteriors.unsqueeze(0)

        _, (rnn_hidden,_) = self.lstm(embeded_posteriors, self.init_hidden())
        rnn_hidden = rnn_hidden.view(self.nlayers, self.ndirections, -1, self.hidden_size)
        quant_embedding = rnn_hidden[0].view(-1)
        quant_embedding = torch.cat((quant_embedding, statistics))

        abstracted = quant_embedding.unsqueeze(0)
        for linear in self.ff_layers:
            abstracted = self.dropout(relu(linear(abstracted)))

        logits = self.output(abstracted).view(1, -1)
        prevalence = torch.softmax(logits, -1)

        return prevalence



