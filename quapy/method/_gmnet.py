"""
GMNet implementation.

Ported from the reference implementation at https://github.com/pglez84/gmnet (the `GMNet`/
`DLQuantification` classes in that repo), adapted to QuaPy's own protocol-based sample generation
(replacing that repo's custom, `quantificationlib`-backed bag generators), and reusing the shared
bag-based training loop already factored out for :class:`quapy.method._histnet.HistNetQ` (see
:class:`quapy.method._neural_bags.BagTrainedQuantifier`).

The overall architecture is: one or more "GM branches" -- each a small per-branch feature extractor
followed by a layer of Gaussian likelihoods (a :class:`_GMLayer`) evaluated at every instance of a bag
-- concatenated and mean-pooled over the bag, followed by the shared quantification MLP head. Like
HistNetQ (and QuaNet), GMNet is trained end-to-end by minimizing a quantification loss over samples
("bags") of known prevalence, rather than over individually labeled instances.

Two deliberate deviations from the reference implementation, both required for the model to satisfy
QuaPy's `predict(X)` contract (i.e., to be usable on a real test collection of arbitrary size, as
opposed to only on bags resampled at the fixed `bag_size` used for training):

* the original `GMNet_Module` reshapes each branch's per-instance likelihoods around a *fixed*,
  constructor-time `bag_size` (via `torch.nn.Unflatten(0, (-1, bag_size))`), which only works when
  every forward pass is fed bags of exactly that size. Here, the reshape is instead computed from the
  actual input shape at forward time (see :class:`_GMBranch`), which is equivalent when the bag size
  matches but also supports bags (or, at prediction time, whole test samples) of any other size.
* the forward hooks used by the original code to capture each branch's pre-Gaussian latent activations
  (for the CKA regularization term) are replaced by simply storing that activation as an attribute
  during `forward` (see :attr:`_GMBranch.latent_activation`), since branches are now implemented with a
  plain `forward` method rather than an opaque `torch.nn.Sequential`.
"""
import numpy as np
import scipy.spatial.distance
import torch
import torch.nn as nn

import geotorch

from quapy.method._neural_bags import BagTrainedQuantifier
from quapy.protocol import UPP


def _cka(latent_activations):
    """Feature-space linear CKA (Centered Kernel Alignment), averaged over every pair of latent
    activations, following the `CKARegularization` class in the reference implementation. Used to
    encourage the Gaussian components learned by different GM branches to capture complementary
    (dissimilar) aspects of the instances.

    :param latent_activations: a list of tensors, one per GM branch, all of shape (n_instances, dim_i)
        (dim_i may differ across branches)
    """
    cka_sum = 0.
    n_pairs = 0
    for i in range(len(latent_activations)):
        for j in range(i + 1, len(latent_activations)):
            x = latent_activations[i]
            y = latent_activations[j]
            x = x - torch.mean(x, dim=0, keepdim=True)
            y = y - torch.mean(y, dim=0, keepdim=True)
            dot_product_similarity = torch.norm(torch.matmul(x.t(), y)) ** 2
            normalization_x = torch.norm(torch.matmul(x.t(), x))
            normalization_y = torch.norm(torch.matmul(y.t(), y))
            cka_sum = cka_sum + dot_product_similarity / (normalization_x * normalization_y)
            n_pairs += 1
    return cka_sum / n_pairs


class _GMLayer(nn.Module):
    """A layer of `num_gaussians` (unnormalized) Gaussian likelihoods, evaluated at every instance of
    a bag. `centers` and `covariance` are learned; `covariance` is constrained to stay positive-definite
    throughout training via `geotorch.positive_definite`.
    """

    def __init__(self, n_features, num_gaussians):
        super().__init__()
        self.n_features = n_features
        self.num_gaussians = num_gaussians
        self.centers = nn.Parameter(torch.rand(num_gaussians, n_features))
        self.covariance = nn.Parameter(torch.eye(n_features).repeat(num_gaussians, 1, 1))
        geotorch.positive_definite(self, "covariance")

        # initialize the centers' covariance from the (squared, halved) nearest-neighbor distance
        # between the randomly initialized centers, so that gaussians start with a sensible spread
        centers = self.centers.detach().cpu().numpy()
        distances = scipy.spatial.distance.cdist(centers, centers)
        np.fill_diagonal(distances, np.inf)
        cov = (np.mean(np.min(distances, axis=1)) / 2) ** 2
        self.covariance = torch.eye(n_features).repeat(num_gaussians, 1, 1) * cov

    def forward(self, x):
        # x: (batch_size, bag_size, n_features)
        centers = self.centers.unsqueeze(0).unsqueeze(0)  # (1, 1, num_gaussians, n_features)
        diff = x.unsqueeze(2) - centers  # (batch_size, bag_size, num_gaussians, n_features)

        cov_inv = torch.inverse(self.covariance)
        det_cov = torch.linalg.det(self.covariance)

        mahalanobis = torch.einsum('...i,...ij,...j->...', diff, cov_inv.unsqueeze(0).unsqueeze(0), diff)
        normalization_term = torch.log((2 * torch.pi) ** self.n_features * det_cov).unsqueeze(0).unsqueeze(0)
        log_probs = -0.5 * (mahalanobis + normalization_term)
        return torch.exp(log_probs)  # (batch_size, bag_size, num_gaussians)


class _GMBranch(nn.Module):
    """One GM branch: an optional small MLP mapping the (already feature-extracted) instances into a
    `gaussian_dimensions`-sized latent space, followed by a Sigmoid, a :class:`_GMLayer`, and a
    BatchNorm applied instance-wise (i.e., over the merged batch*bag_size dimension, matching the
    reference implementation).
    """

    def __init__(self, input_size, num_gaussians, gaussian_dimensions, hidden_size_fe, dropout_fe):
        super().__init__()
        self.pre = nn.Sequential()
        prev_size = input_size
        latent_size = gaussian_dimensions if gaussian_dimensions is not None else input_size
        if gaussian_dimensions is not None:
            for j, layer_size in enumerate(hidden_size_fe or ()):
                self.pre.add_module(f'hidden_{j}', nn.Linear(prev_size, layer_size))
                self.pre.add_module(f'leakyrelu_{j}', nn.LeakyReLU())
                self.pre.add_module(f'dropout_{j}', nn.Dropout(dropout_fe))
                prev_size = layer_size
            self.pre.add_module('latent_linear', nn.Linear(prev_size, gaussian_dimensions))
        self.pre.add_module('sigmoid', nn.Sigmoid())

        self.gm_layer = _GMLayer(n_features=latent_size, num_gaussians=num_gaussians)
        self.batch_norm = nn.BatchNorm1d(num_features=num_gaussians)
        self.output_size = num_gaussians
        self.latent_activation = None  # populated on every forward(), read by GMNet's CKA regularization

    def forward(self, x):
        # x: (batch_size, bag_size, input_size)
        batch_size, bag_size = x.shape[0], x.shape[1]
        latent = self.pre(x)
        self.latent_activation = latent.reshape(-1, latent.shape[-1])
        likelihoods = self.gm_layer(latent)  # (batch_size, bag_size, num_gaussians)
        flat = self.batch_norm(likelihoods.reshape(batch_size * bag_size, -1))
        return flat.reshape(batch_size, bag_size, -1)


class _GMNetModule(nn.Module):
    """The quantification module for GMNet: one or more :class:`_GMBranch` instances, each producing a
    per-instance representation that is concatenated across branches and mean-pooled over the bag, as
    required by :class:`quapy.method._neural_bags.BagTrainedQuantifier`.
    """

    def __init__(self, input_size, num_gaussians, n_gm_layers, gaussian_dimensions, hidden_size_fe=None,
                 dropout_fe=0., cka_regularization=0.):
        super().__init__()
        if len(num_gaussians) != n_gm_layers:
            raise ValueError('num_gaussians should be a tuple of the same size as n_gm_layers')
        if len(gaussian_dimensions) != n_gm_layers:
            raise ValueError('gaussian_dimensions should be a tuple of the same size as n_gm_layers')

        self.n_gm_layers = n_gm_layers
        self.cka_regularization = cka_regularization
        self.branches = nn.ModuleList([
            _GMBranch(input_size, num_gaussians[i], gaussian_dimensions[i], hidden_size_fe, dropout_fe)
            for i in range(n_gm_layers)
        ])
        self.output_size = sum(num_gaussians)

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        return torch.mean(torch.cat(outputs, dim=-1), dim=1)

    def apply_regularization(self):
        """Whether the CKA regularization term should be added to the training loss: requires at least
        two GM branches (CKA is a pairwise measure) and a nonzero `cka_regularization` weight."""
        return self.n_gm_layers > 1 and self.cka_regularization != 0

    def regularization_term(self):
        latent_activations = [branch.latent_activation for branch in self.branches]
        return self.cka_regularization * _cka(latent_activations)


class GMNet(BagTrainedQuantifier):
    """
    Implementation of `GMNet <https://github.com/pglez84/gmnet>`_, a neural network for quantification
    that represents each instance of a bag by its likelihood under one or more learned mixtures of
    Gaussians, mean-pools these representations over the bag, and predicts the class prevalence from the
    result, trained end-to-end by minimizing a quantification loss over many samples ("bags") of known
    prevalence.

    Like :class:`quapy.method._histnet.HistNetQ` and :class:`quapy.method.meta.QuaNet`, GMNet does not
    follow the classify-then-aggregate pattern of :class:`quapy.method.aggregative.AggregativeQuantifier`;
    it is instead trained and evaluated end-to-end on whole bags (see
    :class:`quapy.method._neural_bags.BagTrainedQuantifier` for the shared training/prediction logic,
    including the two entry points, :meth:`fit` and :meth:`fit_from_samples`).

    :param feature_extraction_module: a `torch.nn.Module` exposing an `output_size` attribute, used to
        embed each instance before it is passed to every GM branch. If None (default), an identity
        module is used, i.e., the instances in `X` are assumed to already be in their final numeric
        representation.
    :param n_gm_layers: number of GM branches (default 1).
    :param num_gaussians: number of gaussians per branch: either a single int (used for every branch) or
        a tuple/list of `n_gm_layers` ints (default 4).
    :param gaussian_dimensions: dimensionality of the latent space in which each branch's gaussians live:
        either a single int/None (used for every branch) or a tuple/list of `n_gm_layers` int/None
        values. If None for a given branch, that branch's gaussians operate directly on the
        feature-extracted instances, with no extra per-branch projection (default None).
    :param hidden_size_fe: sizes of the hidden layers of the small per-branch MLP that maps the
        feature-extracted instances into the latent space (only used when `gaussian_dimensions` is not
        None for the corresponding branch); default None (no hidden layers, i.e., a single linear
        projection).
    :param dropout_fe: dropout applied after each of the `hidden_size_fe` layers (default 0).
    :param cka_regularization: weight of the CKA regularization term encouraging the different branches'
        latent representations to be dissimilar; only applied when `n_gm_layers > 1` (default 0, i.e.,
        disabled).
    :param linear_sizes: tuple of ints with the sizes of the linear layers used in the shared
        quantification head, after the GM branches (default empty, i.e., only the final classification
        layer is used).
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
        sampling sequence, as well as for the random initialization of the GM branches (default 0).
    :param checkpointdir: directory where the best model found during training is stored (default
        '../checkpoint').
    :param checkpointname: name of the checkpoint file; if None (default), a random name is generated.
    :param verbose: verbosity level; if >0, shows a progress bar with the current losses (default 0).
    """

    def __init__(self,
                 feature_extraction_module=None,
                 n_gm_layers=1,
                 num_gaussians=4,
                 gaussian_dimensions=None,
                 hidden_size_fe=None,
                 dropout_fe=0.,
                 cka_regularization=0.,
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
        super().__init__(
            feature_extraction_module=feature_extraction_module,
            linear_sizes=linear_sizes,
            dropout=dropout,
            output_function=output_function,
            bag_size=bag_size,
            n_bags_train=n_bags_train,
            n_bags_val=n_bags_val,
            train_epochs=train_epochs,
            patience=patience,
            start_lr=start_lr,
            end_lr=end_lr,
            lr_factor=lr_factor,
            weight_decay=weight_decay,
            quant_loss=quant_loss,
            batch_size=batch_size,
            protocol=protocol,
            protocol_params=protocol_params,
            val_split=val_split,
            device=device,
            random_state=random_state,
            checkpointdir=checkpointdir,
            checkpointname=checkpointname,
            verbose=verbose,
        )
        self.n_gm_layers = n_gm_layers
        self.num_gaussians = num_gaussians if isinstance(num_gaussians, (tuple, list)) \
            else [num_gaussians] * n_gm_layers
        self.gaussian_dimensions = gaussian_dimensions if isinstance(gaussian_dimensions, (tuple, list)) \
            else [gaussian_dimensions] * n_gm_layers
        self.hidden_size_fe = hidden_size_fe
        self.dropout_fe = dropout_fe
        self.cka_regularization = cka_regularization

    @property
    def _checkpoint_prefix(self):
        return 'GMNet'

    def _build_quantmodule(self, n_features):
        torch.manual_seed(self.random_state)
        return _GMNetModule(
            input_size=n_features,
            num_gaussians=self.num_gaussians,
            n_gm_layers=self.n_gm_layers,
            gaussian_dimensions=self.gaussian_dimensions,
            hidden_size_fe=self.hidden_size_fe,
            dropout_fe=self.dropout_fe,
            cka_regularization=self.cka_regularization,
        )

    def _extra_loss(self):
        quantmodule = self.model.quantmodule
        if quantmodule.apply_regularization():
            return quantmodule.regularization_term()
        return 0.
