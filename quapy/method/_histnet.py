"""
HistNetQ implementation, see the original paper:

Pérez-Mon, O., Moreo, A., Coz, JJ del, & González, P. (2025).
Quantification using permutation-invariant networks based on histograms.
Neural Computing and Applications, 37(5), 3505-3520.

Ported from the reference implementation at https://github.com/pglez84/histnetq (the `HistNet`/
`DLQuantification` classes in that repo), adapted to QuaPy's own protocol-based sample generation
(replacing that repo's custom, `quantificationlib`-backed bag generators) and restricted, for now, to
the "hard" differentiable histogram variant by:

Yusuf, I., Igwegbe, G., and Azeez, O. "Differentiable Histogram with Hard-Binning."
arXiv preprint arXiv:2012.06311 (2020).

The overall architecture is: feature_extraction -> Sigmoid -> histogram layer -> small MLP -> softmax,
trained by minimizing a quantification loss over samples ("bags") of known prevalence, rather than
over individually labeled instances (in the spirit of QuaNet, see method/_quanet.py). The bag-based
training loop itself (bag generation, early stopping, LR scheduling, checkpointing, prediction) is
shared with :class:`quapy.method._gmnet.GMNet` via :class:`quapy.method._neural_bags.BagTrainedQuantifier`.
"""
import torch
import torch.nn as nn

from quapy.method._neural_bags import BagTrainedQuantifier
from quapy.protocol import UPP


class _HardHistogramLayer(nn.Module):
    """
    Differentiable "hard" histogram layer. For each feature channel, computes a soft-binned histogram
    over the instances of a bag, using two grouped 1D convolutions (one modeling the distance to each
    bin center, the other modeling the bin width) followed by a thresholded exponential (approximating
    a hard indicator function) and a mean-pool over the bag dimension.
    """

    def __init__(self, n_features, n_bins=8, quantiles=False):
        super().__init__()
        self.in_channels = n_features
        self.n_bins = n_bins
        self.quantiles = quantiles
        self.output_size = n_bins * n_features

        self.bin_centers_conv = nn.Conv1d(
            self.in_channels, self.n_bins * self.in_channels, kernel_size=1, groups=self.in_channels, bias=True
        )
        self.bin_centers_conv.weight.data.fill_(1)
        self.bin_centers_conv.weight.requires_grad = False

        self.bin_widths_conv = nn.Conv1d(
            self.n_bins * self.in_channels, self.n_bins * self.in_channels, kernel_size=1,
            groups=self.n_bins * self.in_channels, bias=True,
        )
        self.bin_widths_conv.weight.data.fill_(-1)
        self.bin_widths_conv.weight.requires_grad = False

        self.threshold = nn.Threshold(1, 0)

        # bin centers evenly spaced in (0, 1), matching the Sigmoid-squashed feature range
        bin_centers = -1 / self.n_bins * (torch.arange(self.n_bins).float() + 0.5)
        self.bin_centers_conv.bias = nn.Parameter(torch.cat(self.in_channels * [bin_centers]), requires_grad=True)
        bin_width = (1 / (2 * self.n_bins)) + 0.001
        self.bin_widths_conv.bias.data.fill_(bin_width)

    def forward(self, input):
        # input: (batch_size, bag_size, n_features)
        if input.dim() == 2:
            input = input.unsqueeze(0)
        result = torch.empty((input.shape[0], self.output_size), device=input.device)
        # the histogram is computed bag by bag (each bag is a "channel-first" 1D signal of length bag_size)
        for i, bag in enumerate(input):
            x = self.bin_centers_conv(bag.transpose(0, 1).unsqueeze(0))
            x = torch.abs(x)
            x = self.bin_widths_conv(x)
            x = torch.pow(1.01, x)
            x = self.threshold(x)
            x = torch.mean(x, dim=2)
            if self.quantiles:
                x = x.view(-1, self.n_bins).cumsum(dim=1)
            result[i, :] = x.flatten()
        return result


class _SigmoidHistogram(nn.Module):
    """The quantification module for HistNetQ: squashes the (already feature-extracted) instances
    through a Sigmoid and builds a differentiable histogram of them, as required by
    :class:`quapy.method._neural_bags.BagTrainedQuantifier`."""

    def __init__(self, n_features, n_bins=8, quantiles=False):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.histogram = _HardHistogramLayer(n_features=n_features, n_bins=n_bins, quantiles=quantiles)
        self.output_size = self.histogram.output_size

    def forward(self, input):
        return self.histogram(self.sigmoid(input))


class HistNetQ(BagTrainedQuantifier):
    """
    Implementation of `HistNetQ <https://github.com/pglez84/histnetq>`_, a neural network for
    quantification that learns a differentiable histogram-based representation of a sample, trained
    end-to-end by minimizing a quantification loss over many samples ("bags") of known prevalence.
    The method was proposed in `Pérez-Mon, O., Moreo, A., Coz, JJ del, & González, P. (2025).
    Quantification using permutation-invariant networks based on histograms.
    Neural Computing and Applications, 37(5), 3505-3520.
    <https://link.springer.com/article/10.1007/s00521-024-10721-1>`_

    HistNetQ does not follow the classify-then-aggregate pattern of :class:`quapy.method.aggregative.
    AggregativeQuantifier`. Such classical approach is termed asymmetric, in the sense that quantifiers
    learn from labelled instances and perform inference over bags.
    Like :class:`quapy.method.meta.QuaNet`, HistNetQ is trained and evaluated
    end-to-end on whole samples rather than on individually labeled instances, following a symmetric problem setting
    (learning from bags, predicting on bags).

    Training data can be provided in two ways: via :meth:`fit`, from a plain labelled collection; or via
    :meth:`fit_from_samples`, from a protocol that already yields the training bags. See
    :class:`quapy.method._neural_bags.BagTrainedQuantifier` for details.

    :param feature_extraction_module: a `torch.nn.Module` exposing an `output_size` attribute, used to
        embed each instance before computing the histogram (e.g., a small MLP for tabular data, a CNN
        for images). If None (default), an identity module is used, i.e., the instances in `X` are
        assumed to already be in their final numeric representation.
    :param n_bins: number of bins used to build the histogram (default 8).
    :param quantiles: if True, use the cumulative (quantile) version of the histogram (default False).
    :param linear_sizes: tuple of ints with the sizes of the linear layers used after the histogram
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
    :param checkpointname: name of the checkpoint file; if None (default), a random name is generated.
    :param verbose: verbosity level; if >0, shows a progress bar with the current losses (default 0).
    """

    def __init__(self,
                 feature_extraction_module=None,
                 n_bins=8,
                 quantiles=False,
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
        self.n_bins = n_bins
        self.quantiles = quantiles

    @property
    def _checkpoint_prefix(self):
        return 'HistNetQ'

    def _build_quantmodule(self, n_features):
        return _SigmoidHistogram(n_features=n_features, n_bins=self.n_bins, quantiles=self.quantiles)
