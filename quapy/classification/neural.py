import os
from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import quapy as qp
from quapy.data import LabelledCollection
from quapy.util import EarlyStop


class NeuralClassifierTrainer:
    """
    Trains a neural network for text classification.

    :param net: an instance of `TextClassifierNet` implementing the forward pass
    :param lr: learning rate (default 1e-3)
    :param weight_decay: weight decay (default 0)
    :param patience: number of epochs that do not show any improvement in validation
        to wait before applying early stop (default 10)
    :param epochs: maximum number of training epochs (default 200)
    :param batch_size: batch size for training (default 64)
    :param batch_size_test: batch size for test (default 512)
    :param padding_length: maximum number of tokens to consider in a document (default 300)
    :param device: specify 'cpu' (default) or 'cuda' for enabling gpu
    :param checkpointpath: where to store the parameters of the best model found so far
        according to the evaluation in the held-out validation split (default '../checkpoint/classifier_net.dat')
    """

    def __init__(self,
                 net: 'TextClassifierNet',
                 lr=1e-3,
                 weight_decay=0,
                 patience=10,
                 epochs=200,
                 batch_size=64,
                 batch_size_test=512,
                 padding_length=300,
                 device='cpu',
                 checkpointpath='../checkpoint/classifier_net.dat'):

        super().__init__()

        assert isinstance(net, TextClassifierNet), f'net is not an instance of {TextClassifierNet.__name__}'
        self.net = net.to(device)
        self.vocab_size = self.net.vocabulary_size
        self.trainer_hyperparams={
            'lr': lr,
            'weight_decay': weight_decay,
            'patience': patience,
            'epochs': epochs,
            'batch_size': batch_size,
            'batch_size_test': batch_size_test,
            'padding_length': padding_length,
            'device': torch.device(device)
        }
        self.learner_hyperparams = self.net.get_params()
        self.checkpointpath = checkpointpath
        self.classes_ = np.asarray([0, 1])

        print(f'[NeuralNetwork running on {device}]')
        os.makedirs(Path(checkpointpath).parent, exist_ok=True)

    def reset_net_params(self, vocab_size, n_classes):
        """Reinitialize the network parameters

        :param vocab_size: the size of the vocabulary
        :param n_classes: the number of target classes
        """
        self.net = self.net.__class__(vocab_size, n_classes, **self.learner_hyperparams)
        self.net = self.net.to(self.trainer_hyperparams['device'])
        self.net.xavier_uniform()

    def get_params(self):
        """Get hyper-parameters for this estimator

        :return: a dictionary with parameter names mapped to their values
        """
        return {**self.net.get_params(), **self.trainer_hyperparams}

    def set_params(self, **params):
        """Set the parameters of this trainer and the learner it is training.
        In this current version, parameter names for the trainer and learner should
        be disjoint.

        :param params: a `**kwargs` dictionary with the parameters
        """
        trainer_hyperparams = self.trainer_hyperparams
        learner_hyperparams = self.net.get_params()
        for key, val in params.items():
            if key in trainer_hyperparams and key in learner_hyperparams:
                raise ValueError(f'the use of parameter {key} is ambiguous since it can refer to '
                                 f'a parameters of the Trainer or the learner {self.net.__name__}')
            elif key not in trainer_hyperparams and key not in learner_hyperparams:
                raise ValueError(f'parameter {key} is not valid')

            if key in trainer_hyperparams:
                trainer_hyperparams[key] = val
            else:
                learner_hyperparams[key] = val

        self.trainer_hyperparams = trainer_hyperparams
        self.learner_hyperparams = learner_hyperparams 

    @property
    def device(self):
        """ Gets the device in which the network is allocated

        :return: device
        """
        return next(self.net.parameters()).device

    def _train_epoch(self, data, status, pbar, epoch):
        self.net.train()
        criterion = torch.nn.CrossEntropyLoss()
        losses, predictions, true_labels = [], [], []
        for xi, yi in data:
            self.optim.zero_grad()
            logits = self.net.forward(xi)
            loss = criterion(logits, yi)
            loss.backward()
            self.optim.step()
            losses.append(loss.item())
            preds = torch.softmax(logits, dim=-1).detach().cpu().numpy().argmax(axis=-1)

            status["loss"] = np.mean(losses)
            predictions.extend(preds.tolist())
            true_labels.extend(yi.detach().cpu().numpy().tolist())
            status["acc"] = accuracy_score(true_labels, predictions)
            status["f1"] = f1_score(true_labels, predictions, average='macro')
            self.__update_progress_bar(pbar, epoch)

    def _test_epoch(self, data, status, pbar, epoch):
        self.net.eval()
        criterion = torch.nn.CrossEntropyLoss()
        losses, predictions, true_labels = [], [], []
        with torch.no_grad():
            for xi, yi in data:
                logits = self.net.forward(xi)
                loss = criterion(logits, yi)
                losses.append(loss.item())
                preds = torch.softmax(logits, dim=-1).detach().cpu().numpy().argmax(axis=-1)
                predictions.extend(preds.tolist())
                true_labels.extend(yi.detach().cpu().numpy().tolist())

            status["loss"] = np.mean(losses)
            status["acc"] = accuracy_score(true_labels, predictions)
            status["f1"] = f1_score(true_labels, predictions, average='macro')
            self.__update_progress_bar(pbar, epoch)

    def __update_progress_bar(self, pbar, epoch):
        pbar.set_description(f'[{self.net.__class__.__name__}] training epoch={epoch} '
                             f'tr-loss={self.status["tr"]["loss"]:.5f} '
                             f'tr-acc={100 * self.status["tr"]["acc"]:.2f}% '
                             f'tr-macroF1={100 * self.status["tr"]["f1"]:.2f}% '
                             f'patience={self.early_stop.patience}/{self.early_stop.PATIENCE_LIMIT} '
                             f'val-loss={self.status["va"]["loss"]:.5f} '
                             f'val-acc={100 * self.status["va"]["acc"]:.2f}% '
                             f'macroF1={100 * self.status["va"]["f1"]:.2f}%')

    def fit(self, instances, labels, val_split=0.3):
        """
        Fits the model according to the given training data.

        :param instances: list of lists of indexed tokens
        :param labels: array-like of shape `(n_samples, n_classes)` with the class labels
        :param val_split: proportion of training documents to be taken as the validation set (default 0.3)
        :return:
        """
        train, val = LabelledCollection(instances, labels).split_stratified(1-val_split)
        opt = self.trainer_hyperparams
        checkpoint = self.checkpointpath
        self.reset_net_params(self.vocab_size, train.n_classes)

        train_generator = TorchDataset(train.instances, train.labels).asDataloader(
            opt['batch_size'], shuffle=True, pad_length=opt['padding_length'], device=opt['device'])
        valid_generator = TorchDataset(val.instances, val.labels).asDataloader(
            opt['batch_size_test'], shuffle=False, pad_length=opt['padding_length'], device=opt['device'])

        self.status = {'tr': {'loss': -1, 'acc': -1, 'f1': -1},
                       'va': {'loss': -1, 'acc': -1, 'f1': -1}}

        self.optim = torch.optim.Adam(self.net.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
        self.early_stop = EarlyStop(opt['patience'], lower_is_better=False)

        with tqdm(range(1, opt['epochs'] + 1)) as pbar:
            for epoch in pbar:
                self._train_epoch(train_generator, self.status['tr'], pbar, epoch)
                self._test_epoch(valid_generator, self.status['va'], pbar, epoch)

                self.early_stop(self.status['va']['f1'], epoch)
                if self.early_stop.IMPROVED:
                    torch.save(self.net.state_dict(), checkpoint)
                elif self.early_stop.STOP:
                    print(f'training ended by patience exhasted; loading best model parameters in {checkpoint} '
                          f'for epoch {self.early_stop.best_epoch}')
                    self.net.load_state_dict(torch.load(checkpoint))
                    break

        print('performing one training pass over the validation set...')
        self._train_epoch(valid_generator, self.status['tr'], pbar, epoch=0)
        print('[done]')

        return self

    def predict(self, instances):
        """
        Predicts labels for the instances

        :param instances: list of lists of indexed tokens
        :return: a `numpy` array of length `n` containing the label predictions, where `n` is the number of
            instances in `X`
        """
        return np.argmax(self.predict_proba(instances), axis=-1)

    def predict_proba(self, instances):
        """
        Predicts posterior probabilities for the instances

        :param X: array-like of shape `(n_samples, n_features)` instances to classify
        :return: array-like of shape `(n_samples, n_classes)` with the posterior probabilities
        """
        self.net.eval()
        opt = self.trainer_hyperparams
        with torch.no_grad():
            positive_probs = []
            for xi in TorchDataset(instances).asDataloader(
                    opt['batch_size_test'], shuffle=False, pad_length=opt['padding_length'], device=opt['device']):
                positive_probs.append(self.net.predict_proba(xi))
        return np.concatenate(positive_probs)

    def transform(self, instances):
        """
        Returns the embeddings of the instances

        :param instances: list of lists of indexed tokens
        :return: array-like of shape `(n_samples, embed_size)` with the embedded instances,
            where `embed_size` is defined by the classification network
        """
        self.net.eval()
        embeddings = []
        opt = self.trainer_hyperparams
        with torch.no_grad():
            for xi in TorchDataset(instances).asDataloader(
                    opt['batch_size_test'], shuffle=False, pad_length=opt['padding_length'], device=opt['device']):
                embeddings.append(self.net.document_embedding(xi).detach().cpu().numpy())
        return np.concatenate(embeddings)


class TorchDataset(torch.utils.data.Dataset):
    """
    Transforms labelled instances into a Torch's :class:`torch.utils.data.DataLoader` object

    :param instances: list of lists of indexed tokens
    :param labels: array-like of shape `(n_samples, n_classes)` with the class labels
    """

    def __init__(self, instances, labels=None):
        self.instances = instances
        self.labels = labels

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return {'doc': self.instances[index], 'label': self.labels[index] if self.labels is not None else None}

    def asDataloader(self, batch_size, shuffle, pad_length, device):
        """
        Converts the labelled collection into a Torch DataLoader with dynamic padding for
        the batch

        :param batch_size: batch size
        :param shuffle: whether or not to shuffle instances
        :param pad_length: the maximum length for the list of tokens (dynamic padding is
            applied, meaning that if the longest document in the batch is shorter than
            `pad_length`, then the batch is padded up to its length, and not to `pad_length`.
        :param device: whether to allocate tensors in cpu or in cuda
        :return: a :class:`torch.utils.data.DataLoader` object
        """
        def collate(batch):
            data = [torch.LongTensor(item['doc'][:pad_length]) for item in batch]
            data = pad_sequence(data, batch_first=True, padding_value=qp.environ['PAD_INDEX']).to(device)
            targets = [item['label'] for item in batch]
            if targets[0] is None:
                return data
            else:
                targets = torch.as_tensor(targets, dtype=torch.long).to(device)
                return [data, targets]

        torchDataset = TorchDataset(self.instances, self.labels)
        return torch.utils.data.DataLoader(torchDataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


class TextClassifierNet(torch.nn.Module, metaclass=ABCMeta):
    """
    Abstract Text classifier (`torch.nn.Module`)
    """

    @abstractmethod
    def document_embedding(self, x):
        """Embeds documents (i.e., performs the forward pass up to the
        next-to-last layer).

        :param x: a batch of instances, typically generated by a torch's `DataLoader`
            instance (see :class:`quapy.classification.neural.TorchDataset`)
        :return: a torch tensor of shape `(n_samples, n_dimensions)`, where
            `n_samples` is the number of documents, and `n_dimensions` is the
            dimensionality of the embedding
        """
        ...

    def forward(self, x):
        """Performs the forward pass.

        :param x: a batch of instances, typically generated by a torch's `DataLoader`
            instance (see :class:`quapy.classification.neural.TorchDataset`)
        :return: a tensor of shape `(n_instances, n_classes)` with the decision scores
            for each of the instances and classes
        """
        doc_embedded = self.document_embedding(x)
        return self.output(doc_embedded)

    def dimensions(self):
        """Gets the number of dimensions of the embedding space

        :return: integer
        """
        return self.dim

    def predict_proba(self, x):
        """
        Predicts posterior probabilities for the instances in `x`

        :param x: a torch tensor of indexed tokens with shape `(n_instances, pad_length)`
            where `n_instances` is the number of instances in the batch, and `pad_length`
            is length of the pad in the batch
        :return: array-like of shape `(n_samples, n_classes)` with the posterior probabilities
        """
        logits = self(x)
        return torch.softmax(logits, dim=1).detach().cpu().numpy()

    def xavier_uniform(self):
        """
        Performs Xavier initialization of the network parameters
        """
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                torch.nn.init.xavier_uniform_(p)

    @abstractmethod
    def get_params(self):
        """
        Get hyper-parameters for this estimator

        :return: a dictionary with parameter names mapped to their values
        """
        ...

    @property
    def vocabulary_size(self):
        """
        Return the size of the vocabulary

        :return: integer
        """
        ...


class LSTMnet(TextClassifierNet):
    """
    An implementation of :class:`quapy.classification.neural.TextClassifierNet` based on
    Long Short Term Memory networks.

    :param vocabulary_size: the size of the vocabulary
    :param n_classes: number of target classes
    :param embedding_size: the dimensionality of the word embeddings space (default 100)
    :param hidden_size: the dimensionality of the hidden space (default 256)
    :param repr_size: the dimensionality of the document embeddings space (default 100)
    :param lstm_class_nlayers: number of LSTM layers (default 1)
    :param drop_p: drop probability for dropout (default 0.5)
    """

    def __init__(self, vocabulary_size, n_classes, embedding_size=100, hidden_size=256, repr_size=100, lstm_class_nlayers=1,
                 drop_p=0.5):

        super().__init__()
        self.vocabulary_size_ = vocabulary_size
        self.n_classes = n_classes
        self.hyperparams={
            'embedding_size': embedding_size,
            'hidden_size': hidden_size,
            'repr_size': repr_size,
            'lstm_class_nlayers': lstm_class_nlayers,
            'drop_p': drop_p
        }

        self.word_embedding = torch.nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = torch.nn.LSTM(embedding_size, hidden_size, lstm_class_nlayers, dropout=drop_p, batch_first=True)
        self.dropout = torch.nn.Dropout(drop_p)

        self.dim = repr_size
        self.doc_embedder = torch.nn.Linear(hidden_size, self.dim)
        self.output = torch.nn.Linear(self.dim, n_classes)

    def __init_hidden(self, set_size):
        opt = self.hyperparams
        var_hidden = torch.zeros(opt['lstm_class_nlayers'], set_size, opt['hidden_size'])
        var_cell = torch.zeros(opt['lstm_class_nlayers'], set_size, opt['hidden_size'])
        if next(self.lstm.parameters()).is_cuda:
            var_hidden, var_cell = var_hidden.cuda(), var_cell.cuda()
        return var_hidden, var_cell

    def document_embedding(self, x):
        """Embeds documents (i.e., performs the forward pass up to the
        next-to-last layer).

        :param x: a batch of instances, typically generated by a torch's `DataLoader`
            instance (see :class:`quapy.classification.neural.TorchDataset`)
        :return: a torch tensor of shape `(n_samples, n_dimensions)`, where
            `n_samples` is the number of documents, and `n_dimensions` is the
            dimensionality of the embedding
        """
        embedded = self.word_embedding(x)
        rnn_output, rnn_hidden = self.lstm(embedded, self.__init_hidden(x.size()[0]))
        abstracted = self.dropout(F.relu(rnn_hidden[0][-1]))
        abstracted = self.doc_embedder(abstracted)
        return abstracted

    def get_params(self):
        """
        Get hyper-parameters for this estimator

        :return: a dictionary with parameter names mapped to their values
        """
        return self.hyperparams

    @property
    def vocabulary_size(self):
        """
        Return the size of the vocabulary

        :return: integer
        """
        return self.vocabulary_size_


class CNNnet(TextClassifierNet):
    """
    An implementation of :class:`quapy.classification.neural.TextClassifierNet` based on
    Convolutional Neural Networks.

    :param vocabulary_size: the size of the vocabulary
    :param n_classes: number of target classes
    :param embedding_size: the dimensionality of the word embeddings space (default 100)
    :param hidden_size: the dimensionality of the hidden space (default 256)
    :param repr_size: the dimensionality of the document embeddings space (default 100)
    :param kernel_heights: list of kernel lengths (default [3,5,7]), i.e., the number of
        consecutive tokens that each kernel covers
    :param stride: convolutional stride (default 1)
    :param stride: convolutional pad (default 0)
    :param drop_p: drop probability for dropout (default 0.5)
    """

    def __init__(self, vocabulary_size, n_classes, embedding_size=100, hidden_size=256, repr_size=100,
                 kernel_heights=[3, 5, 7], stride=1, padding=0, drop_p=0.5):
        super(CNNnet, self).__init__()

        self.vocabulary_size_ = vocabulary_size
        self.n_classes = n_classes
        self.hyperparams={
            'embedding_size': embedding_size,
            'hidden_size': hidden_size,
            'repr_size': repr_size,
            'kernel_heights':kernel_heights,
            'stride': stride,
            'drop_p': drop_p
        }
        self.word_embedding = torch.nn.Embedding(vocabulary_size, embedding_size)
        in_channels = 1
        self.conv1 = nn.Conv2d(in_channels, hidden_size, (kernel_heights[0], embedding_size), stride, padding)
        self.conv2 = nn.Conv2d(in_channels, hidden_size, (kernel_heights[1], embedding_size), stride, padding)
        self.conv3 = nn.Conv2d(in_channels, hidden_size, (kernel_heights[2], embedding_size), stride, padding)
        self.dropout = nn.Dropout(drop_p)

        self.dim = repr_size
        self.doc_embedder = torch.nn.Linear(len(kernel_heights) * hidden_size, self.dim)
        self.output = nn.Linear(self.dim, n_classes)

    def __conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def document_embedding(self, input):
        """Embeds documents (i.e., performs the forward pass up to the
        next-to-last layer).

        :param input: a batch of instances, typically generated by a torch's `DataLoader`
            instance (see :class:`quapy.classification.neural.TorchDataset`)
        :return: a torch tensor of shape `(n_samples, n_dimensions)`, where
            `n_samples` is the number of documents, and `n_dimensions` is the
            dimensionality of the embedding
        """
        input = self.word_embedding(input)
        input = input.unsqueeze(1)  # input.size() = (batch_size, 1, num_seq, embedding_length)

        max_out1 = self.__conv_block(input, self.conv1)
        max_out2 = self.__conv_block(input, self.conv2)
        max_out3 = self.__conv_block(input, self.conv3)

        all_out = torch.cat((max_out1, max_out2, max_out3), 1)  # all_out.size() = (batch_size, num_kernels*out_channels)
        abstracted = self.dropout(F.relu(all_out))  #  (batch_size, num_kernels*out_channels)
        abstracted = self.doc_embedder(abstracted)
        return abstracted

    def get_params(self):
        """
        Get hyper-parameters for this estimator

        :return: a dictionary with parameter names mapped to their values
        """
        return self.hyperparams

    @property
    def vocabulary_size(self):
        """
        Return the size of the vocabulary

        :return: integer
        """
        return self.vocabulary_size_





