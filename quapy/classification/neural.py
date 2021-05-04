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

    def __init__(self,
                 net,  # TextClassifierNet
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
        self.net = self.net.__class__(vocab_size, n_classes, **self.learner_hyperparams)
        self.net = self.net.to(self.trainer_hyperparams['device'])
        self.net.xavier_uniform()

    def get_params(self):
        return {**self.net.get_params(), **self.trainer_hyperparams}

    def set_params(self, **params):
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
        return np.argmax(self.predict_proba(instances), axis=-1)

    def predict_proba(self, instances):
        self.net.eval()
        opt = self.trainer_hyperparams
        with torch.no_grad():
            positive_probs = []
            for xi in TorchDataset(instances).asDataloader(
                    opt['batch_size_test'], shuffle=False, pad_length=opt['padding_length'], device=opt['device']):
                positive_probs.append(self.net.predict_proba(xi))
        return np.concatenate(positive_probs)

    def transform(self, instances):
        self.net.eval()
        embeddings = []
        opt = self.trainer_hyperparams
        with torch.no_grad():
            for xi in TorchDataset(instances).asDataloader(
                    opt['batch_size_test'], shuffle=False, pad_length=opt['padding_length'], device=opt['device']):
                embeddings.append(self.net.document_embedding(xi).detach().cpu().numpy())
        return np.concatenate(embeddings)


class TorchDataset(torch.utils.data.Dataset):

    def __init__(self, instances, labels=None):
        self.instances = instances
        self.labels = labels

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return {'doc': self.instances[index], 'label': self.labels[index] if self.labels is not None else None}

    def asDataloader(self, batch_size, shuffle, pad_length, device):
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

    @abstractmethod
    def document_embedding(self, x): ...

    def forward(self, x):
        doc_embedded = self.document_embedding(x)
        return self.output(doc_embedded)

    def dimensions(self):
        return self.dim

    def predict_proba(self, x):
        logits = self(x)
        return torch.softmax(logits, dim=1).detach().cpu().numpy()

    def xavier_uniform(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                torch.nn.init.xavier_uniform_(p)

    @abstractmethod
    def get_params(self): ...

    @property
    def vocabulary_size(self): ...


class LSTMnet(TextClassifierNet):

    def __init__(self, vocabulary_size, n_classes, embedding_size=100, hidden_size=256, repr_size=100, lstm_nlayers=1,
                 drop_p=0.5):
        super().__init__()
        self.vocabulary_size_ = vocabulary_size
        self.n_classes = n_classes
        self.hyperparams={
            'embedding_size': embedding_size,
            'hidden_size': hidden_size,
            'repr_size': repr_size,
            'lstm_nlayers': lstm_nlayers,
            'drop_p': drop_p
        }

        self.word_embedding = torch.nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = torch.nn.LSTM(embedding_size, hidden_size, lstm_nlayers, dropout=drop_p, batch_first=True)
        self.dropout = torch.nn.Dropout(drop_p)

        self.dim = repr_size
        self.doc_embedder = torch.nn.Linear(hidden_size, self.dim)
        self.output = torch.nn.Linear(self.dim, n_classes)

    def init_hidden(self, set_size):
        opt = self.hyperparams
        var_hidden = torch.zeros(opt['lstm_nlayers'], set_size, opt['lstm_hidden_size'])
        var_cell = torch.zeros(opt['lstm_nlayers'], set_size, opt['lstm_hidden_size'])
        if next(self.lstm.parameters()).is_cuda:
            var_hidden, var_cell = var_hidden.cuda(), var_cell.cuda()
        return var_hidden, var_cell

    def document_embedding(self, x):
        embedded = self.word_embedding(x)
        rnn_output, rnn_hidden = self.lstm(embedded, self.init_hidden(x.size()[0]))
        abstracted = self.dropout(F.relu(rnn_hidden[0][-1]))
        abstracted = self.doc_embedder(abstracted)
        return abstracted

    def get_params(self):
        return self.hyperparams

    @property
    def vocabulary_size(self):
        return self.vocabulary_size_


class CNNnet(TextClassifierNet):

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

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def document_embedding(self, input):
        input = self.word_embedding(input)
        input = input.unsqueeze(1)  # input.size() = (batch_size, 1, num_seq, embedding_length)

        max_out1 = self.conv_block(input, self.conv1)
        max_out2 = self.conv_block(input, self.conv2)
        max_out3 = self.conv_block(input, self.conv3)

        all_out = torch.cat((max_out1, max_out2, max_out3), 1)  # all_out.size() = (batch_size, num_kernels*out_channels)
        abstracted = self.dropout(F.relu(all_out))  #  (batch_size, num_kernels*out_channels)
        abstracted = self.doc_embedder(abstracted)
        return abstracted

    def get_params(self):
        return self.hyperparams

    @property
    def vocabulary_size(self):
        return self.vocabulary_size_





