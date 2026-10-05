"""
Shared machinery for QuaPy's "bag-trained" neural quantifiers, i.e., methods that -- like
:class:`quapy.method.meta.QuaNet` -- do not follow the classify-then-aggregate pattern of
:class:`quapy.method.aggregative.AggregativeQuantifier`, but are instead trained and evaluated
end-to-end on whole samples ("bags") of known prevalence.

:class:`quapy.method._histnet.HistNetQ` and :class:`quapy.method._gmnet.GMNet` share the same overall
architecture (feature_extraction -> quantification module -> small MLP head -> softmax/normalize) and
the same bag-based training protocol (bag generation via a QuaPy sampling protocol, early stopping, LR
scheduling, checkpointing). This module factors that common part out into :class:`BagTrainedQuantifier`;
concrete subclasses only need to supply the quantification module placed between the feature extractor
and the shared head (see :meth:`BagTrainedQuantifier._build_quantmodule`).
"""
import copy
import os
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier
from quapy.protocol import AbstractProtocol, UPP
from quapy.util import EarlyStop


class IdentityFeatureExtractionModule(nn.Module):
    """Used when no feature extraction module is provided: instances are assumed to already be in
    their final numeric representation."""

    def __init__(self, input_size):
        super().__init__()
        self.output_size = input_size

    def forward(self, x):
        return x


def to_tensor(x, device):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32)
    if hasattr(x, 'toarray'):  # scipy sparse
        x = x.toarray()
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)


def stack_bags(bags, device):
    """
    :param bags: an iterable of (X_bag, prevalence) pairs, all X_bag with the same number of instances
    :return: a pair of tensors (X, P) of shape (n_bags, bag_size, n_features) and (n_bags, n_classes)
    """
    Xs, ps = zip(*bags)
    X = torch.stack([to_tensor(x, device) for x in Xs])
    P = torch.stack([to_tensor(p, device) for p in ps])
    return X, P


def mix_two_bags(bag_a, bag_b, bag_size, rng):
    """Synthesizes a new bag of size `bag_size` by mixing two given bags with a random ratio, following
    the "mixer" idea from the original HistNetQ repo (`UnlabeledMixerBagGenerator`): useful when the
    only available training material is a modest number of pre-built samples (e.g., LeQua's dev
    samples) and one wants extra intermediate-prevalence bags without access to instance-level labels.
    """
    Xa, pa = bag_a
    Xb, pb = bag_b
    m = rng.random()
    na = round(m * bag_size)
    nb = bag_size - na
    idx_a = rng.choices(range(len(Xa)), k=na) if na > 0 else []
    idx_b = rng.choices(range(len(Xb)), k=nb) if nb > 0 else []
    Xa, Xb = np.asarray(Xa), np.asarray(Xb)
    X_mixed = np.concatenate([Xa[idx_a], Xb[idx_b]], axis=0)
    p_mixed = m * np.asarray(pa, dtype=float) + (1 - m) * np.asarray(pb, dtype=float)
    return X_mixed, p_mixed


def build_output_head(input_size, n_classes, linear_sizes, dropout, output_function):
    """Builds the small MLP + output activation shared by every bag-trained quantifier's head: a stack
    of (Linear, LeakyReLU, Dropout) blocks sized by `linear_sizes`, followed by a final Linear to
    `n_classes` and either a softmax or an L1-normalization (applied in :class:`BagNetworkModule`), both
    yielding a valid prevalence vector."""
    output_module = nn.Sequential()
    prev_size = input_size
    for i, linear_size in enumerate(linear_sizes):
        output_module.add_module(f'linear_{i}', nn.Linear(prev_size, linear_size))
        output_module.add_module(f'leakyrelu_{i}', nn.LeakyReLU())
        output_module.add_module(f'dropout_{i}', nn.Dropout(dropout))
        prev_size = linear_size
    output_module.add_module('last_linear', nn.Linear(prev_size, n_classes))
    if output_function == 'softmax':
        output_module.add_module('softmax', nn.Softmax(dim=1))
    elif output_function == 'normalize':
        output_module.add_module('relu', nn.ReLU())
    else:
        raise ValueError(f"unknown {output_function=}; valid ones are 'softmax', 'normalize'")
    return output_module


class BagNetworkModule(nn.Module):
    """The full network shared by every bag-trained quantifier: feature extraction, a pluggable
    quantification module (mapping a bag of instance-level features to a single per-bag
    representation), and the shared MLP head.

    :param quantmodule: a `torch.nn.Module` exposing an `output_size` attribute, mapping a tensor of
        shape (batch_size, bag_size, n_features) to one of shape (batch_size, quantmodule.output_size).
    """

    def __init__(self, feature_extraction_module, quantmodule, n_classes, linear_sizes, dropout, output_function):
        super().__init__()
        self.feature_extraction_module = feature_extraction_module
        self.quantmodule = quantmodule
        self.output_function = output_function
        self.output_module = build_output_head(
            quantmodule.output_size, n_classes, linear_sizes, dropout, output_function
        )

    def forward(self, bag):
        # bag: (batch_size, bag_size, n_features)
        features = self.feature_extraction_module(bag)
        representation = self.quantmodule(features)
        out = self.output_module(representation)
        if self.output_function == 'normalize':
            out = nn.functional.normalize(out, p=1, dim=1)
        return out


class BagTrainedQuantifier(BaseQuantifier, ABC):
    """
    Base class for QuaPy's neural quantifiers trained end-to-end on whole samples ("bags") of known
    prevalence, rather than following the classify-then-aggregate pattern of
    :class:`quapy.method.aggregative.AggregativeQuantifier` (in the spirit of
    :class:`quapy.method.meta.QuaNet`). Concrete subclasses only need to provide the quantification
    module placed between the feature extractor and the shared MLP head (see
    :meth:`_build_quantmodule`) and a checkpoint-name prefix (see :attr:`_checkpoint_prefix`); bag
    generation, the training/validation loop, early stopping, LR scheduling, checkpointing, and
    prediction are all shared.

    Training data can be provided in two ways:

    * via :meth:`fit`, from a plain labelled collection (`X`, `y`): training/validation bags are then
      generated by resampling from it using a QuaPy sampling protocol (:class:`quapy.protocol.UPP` by
      default).
    * via :meth:`fit_from_samples`, from a :class:`quapy.protocol.AbstractProtocol` that already yields
      the training bags (e.g., :class:`quapy.data._lequa.SamplesFromDir` for LeQua-style pre-built
      samples), optionally enriched with synthetic bags mixed from the given ones.

    :param feature_extraction_module: a `torch.nn.Module` exposing an `output_size` attribute, used to
        embed each instance before the quantification module (e.g., a small MLP for tabular data, a CNN
        for images). If None (default), an identity module is used, i.e., the instances in `X` are
        assumed to already be in their final numeric representation.
    :param linear_sizes: tuple of ints with the sizes of the linear layers used in the shared head
        (default empty, i.e., only the final classification layer is used).
    :param dropout: dropout applied after each of the `linear_sizes` layers (default 0).
    :param output_function: either 'softmax' or 'normalize' (L1); both yield a valid prevalence vector
        (default 'softmax').
    :param bag_size: number of instances per training/validation bag (default 500).
    :param n_bags_train: number of bags generated per training epoch (default 500).
    :param n_bags_val: number of bags generated per validation epoch (default 500).
    :param train_epochs: maximum number of training epochs (default 200).
    :param patience: number of epochs without improvement in validation loss before early-stopping
        (default 20).
    :param start_lr: initial learning rate (default 1e-3).
    :param end_lr: once the learning rate decays below this value, training stops (default 1e-6).
    :param lr_factor: factor by which the learning rate is reduced after `patience` epochs without
        improvement (default 0.1).
    :param weight_decay: L2 regularization (default 0).
    :param quant_loss: the quantification loss to minimize (default `torch.nn.L1Loss()`), called as
        `quant_loss(true_prevalences, predicted_prevalences)`.
    :param batch_size: number of bags per gradient update (default 16).
    :param protocol: the :class:`quapy.protocol.AbstractStochasticSeededProtocol` subclass used by
        :meth:`fit` to resample bags from the given labelled collection (default
        :class:`quapy.protocol.UPP`, which draws bags with prevalence sampled uniformly at random from
        the simplex).
    :param protocol_params: dict of extra keyword arguments passed to `protocol` (besides `data`,
        `sample_size`, `repeats`, and `random_state`, which are set internally); default None.
    :param val_split: float in (0,1), the proportion of the collection given to :meth:`fit` that is held
        out (via stratified sampling) for validation and early stopping (default 0.4).
    :param device: `'cpu'` or `'cuda'` (default 'cpu').
    :param random_state: seed used for the train/validation split and for the (fixed) validation
        sampling sequence (default 0).
    :param checkpointdir: directory where the best model found during training is stored (default
        '../checkpoint').
    :param checkpointname: name of the checkpoint file; if None (default), a random name prefixed by
        :attr:`_checkpoint_prefix` is generated.
    :param verbose: verbosity level; if >0, shows a progress bar with the current losses (default 0).
    """

    def __init__(self,
                 feature_extraction_module=None,
                 linear_sizes=(),
                 dropout=0.,
                 output_function='softmax',
                 bag_size=500,
                 n_bags_train=500,
                 n_bags_val=500,
                 train_epochs=200,
                 patience=20,
                 start_lr=1e-3,
                 end_lr=1e-6,
                 lr_factor=0.1,
                 weight_decay=0.,
                 quant_loss=None,
                 batch_size=16,
                 protocol=UPP,
                 protocol_params=None,
                 val_split=0.4,
                 device='cpu',
                 random_state=0,
                 checkpointdir='../checkpoint',
                 checkpointname=None,
                 verbose=0):
        self.feature_extraction_module = feature_extraction_module
        self.linear_sizes = linear_sizes
        self.dropout = dropout
        self.output_function = output_function
        self.bag_size = bag_size
        self.n_bags_train = n_bags_train
        self.n_bags_val = n_bags_val
        self.train_epochs = train_epochs
        self.patience = patience
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_factor = lr_factor
        self.weight_decay = weight_decay
        self.quant_loss = quant_loss if quant_loss is not None else torch.nn.L1Loss()
        self.batch_size = batch_size
        self.protocol = protocol
        self.protocol_params = protocol_params
        self.val_split = val_split
        self.device = torch.device(device)
        self.random_state = random_state
        if checkpointname is None:
            local_random = random.Random()
            random_code = '-'.join(str(local_random.randint(0, 1000000)) for _ in range(5))
            checkpointname = f'{self._checkpoint_prefix}-{random_code}'
        self.checkpointdir = checkpointdir
        self.checkpoint = os.path.join(checkpointdir, checkpointname)
        self.verbose = verbose
        self._classes_ = None
        self.model = None

    @property
    def classes_(self):
        return self._classes_

    @property
    @abstractmethod
    def _checkpoint_prefix(self):
        """Short name used as the default checkpoint filename prefix (e.g. 'HistNetQ', 'GMNet')."""
        ...

    @abstractmethod
    def _build_quantmodule(self, n_features):
        """Builds the module placed between the (already feature-extracted) instances and the shared
        MLP head. Must expose an `output_size` attribute and accept input of shape
        (batch_size, bag_size, n_features), returning one of shape (batch_size, output_size)."""
        ...

    def _extra_loss(self):
        """Optional additional term added to the quantification loss during training (e.g., GMNet's CKA
        regularization across GM layers). Returns 0 by default."""
        return 0.

    def fit(self, X, y):
        """
        Trains the quantifier from a plain labelled collection, generating training and validation bags
        by resampling from it via `self.protocol` (a fresh random sequence of bags every epoch for
        training, and a fixed, reproducible sequence for validation).

        :param X: the training instances
        :param y: the labels of X
        :return: self
        """
        data = LabelledCollection(X, y)
        self._classes_ = data.classes_
        train_data, val_data = data.split_stratified(train_prop=1 - self.val_split, random_state=self.random_state)

        protocol_params = self.protocol_params or {}

        def train_bags():
            sampler = self.protocol(
                train_data, sample_size=self.bag_size, repeats=self.n_bags_train, random_state=None,
                **protocol_params
            )
            return sampler()

        def val_bags():
            sampler = self.protocol(
                val_data, sample_size=self.bag_size, repeats=self.n_bags_val, random_state=self.random_state,
                **protocol_params
            )
            return sampler()

        n_features = train_data.instances.shape[1]
        self._fit_loop(train_bags, val_bags, n_features, n_bags_train=self.n_bags_train, n_bags_val=self.n_bags_val)
        return self

    def fit_from_samples(self, protocol: AbstractProtocol, val_protocol: AbstractProtocol = None,
                          mix_bags=False, mix_bags_proportion=0.5):
        """
        Trains the quantifier from a protocol that already yields the training bags (e.g.,
        :class:`quapy.data._lequa.SamplesFromDir`, for LeQua-style pre-built samples), instead of
        resampling from a labelled collection. This is the entry point to use whenever only bags of
        known prevalence are available (no instance-level labels).

        :param protocol: an :class:`AbstractProtocol` yielding `(sample, prevalence)` pairs; consumed
            once and kept in memory (expected to be of modest size, as is typical of pre-built sample
            collections).
        :param val_protocol: an optional, separate protocol providing the validation bags; if None, a
            `val_split` fraction of the bags returned by `protocol` is held out instead.
        :param mix_bags: if True, in addition to the bags returned by `protocol`, synthesize extra bags
            each epoch by mixing random pairs of the given bags with a random ratio (a substitute for
            the original HistNetQ repo's `UnlabeledMixerBagGenerator`, useful to broaden the coverage of
            prevalence values beyond what the given bags exhibit).
        :param mix_bags_proportion: proportion (relative to the number of base training bags) of extra
            mixed bags to generate per epoch when `mix_bags=True` (default 0.5).
        :return: self
        """
        assert isinstance(protocol, AbstractProtocol), 'protocol must be an instance of AbstractProtocol'
        base_bags = list(protocol())
        n_classes = len(np.asarray(base_bags[0][1]))
        self._classes_ = np.arange(n_classes)

        if val_protocol is not None:
            val_bags_list = list(val_protocol())
        else:
            n_val = max(1, int(len(base_bags) * self.val_split))
            val_bags_list = base_bags[:n_val]
            base_bags = base_bags[n_val:]

        rng = random.Random(self.random_state)
        n_mixed = round(len(base_bags) * mix_bags_proportion) if mix_bags else 0

        def train_bags():
            bags = list(base_bags)
            if n_mixed > 0:
                for _ in range(n_mixed):
                    a, b = rng.choice(base_bags), rng.choice(base_bags)
                    bags.append(mix_two_bags(a, b, self.bag_size, rng))
            rng.shuffle(bags)
            return bags

        def val_bags():
            return val_bags_list

        n_features = np.asarray(base_bags[0][0]).shape[1]
        self._fit_loop(
            train_bags, val_bags, n_features,
            n_bags_train=len(base_bags) + n_mixed, n_bags_val=len(val_bags_list)
        )
        return self

    def _fit_loop(self, train_bags_fn, val_bags_fn, n_features, n_bags_train, n_bags_val):
        os.makedirs(self.checkpointdir, exist_ok=True)
        n_classes = len(self._classes_)

        fe = self.feature_extraction_module
        if fe is None:
            fe = IdentityFeatureExtractionModule(n_features)
        quantmodule = self._build_quantmodule(fe.output_size)
        self.model = BagNetworkModule(
            fe, quantmodule, n_classes, linear_sizes=self.linear_sizes, dropout=self.dropout,
            output_function=self.output_function
        ).to(self.device)

        optim = torch.optim.Adam(self.model.parameters(), lr=self.start_lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=self.patience, factor=self.lr_factor)
        early_stop = EarlyStop(self.patience, lower_is_better=True)

        best_state = copy.deepcopy(self.model.state_dict())
        for epoch in range(self.train_epochs):
            self._run_epoch(train_bags_fn(), n_bags_train, optim, train=True, epoch=epoch)
            va_loss = self._run_epoch(val_bags_fn(), n_bags_val, optim=None, train=False, epoch=epoch)

            early_stop(va_loss, epoch)
            if early_stop.IMPROVED:
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save(best_state, self.checkpoint)
            elif early_stop.STOP:
                if self.verbose > 0:
                    print(f'[{self._checkpoint_prefix}] training ended by patience exhausted at epoch {epoch}; '
                          f'restoring best model from epoch {early_stop.best_epoch}')
                break

            scheduler.step(va_loss)
            if optim.param_groups[0]['lr'] < self.end_lr:
                if self.verbose > 0:
                    print(f'[{self._checkpoint_prefix}] early stopping in epoch {epoch} (learning rate below end_lr)')
                break

        self.model.load_state_dict(best_state)

    def _run_epoch(self, bags, n_bags, optim, train, epoch):
        self.model.train(mode=train)
        losses = []
        pbar = tqdm(bags, total=n_bags, disable=self.verbose == 0)
        batch = []

        def process_batch(batch):
            X, P = stack_bags(batch, self.device)
            if train:
                optim.zero_grad()
                P_hat = self.model.forward(X)
                loss = self.quant_loss(P, P_hat) + self._extra_loss()
                loss.backward()
                optim.step()
            else:
                with torch.no_grad():
                    P_hat = self.model.forward(X)
                    loss = self.quant_loss(P, P_hat)
            return loss.item()

        for bag in pbar:
            batch.append(bag)
            if len(batch) == self.batch_size:
                losses.append(process_batch(batch))
                batch = []
                pbar.set_description(
                    f'[{self._checkpoint_prefix}] epoch={epoch} {"train" if train else "val"}-'
                    f'loss={np.mean(losses):.5f}'
                )
        if batch:
            losses.append(process_batch(batch))

        return np.mean(losses) if losses else float('inf')

    def predict(self, X):
        """
        Generates a class prevalence estimate for the sample `X`, via a single forward pass of the
        trained network (the quantification module aggregates over however many instances are given, so
        `X` need not match the `bag_size` used during training).

        :param X: the test instances
        :return: `np.ndarray` of shape `(n_classes,)` with the class prevalence estimates
        """
        self.model.eval()
        with torch.no_grad():
            X_t = to_tensor(X, self.device).unsqueeze(0)
            prevalence = self.model.forward(X_t)
            return prevalence.cpu().numpy().flatten()
