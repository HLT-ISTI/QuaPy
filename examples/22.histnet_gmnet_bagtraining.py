import quapy as qp
from quapy.data.datasets import fetch_image_embeddings
from quapy.method.meta import HistNetQ, GMNet
from quapy.protocol import UPP

# This example showcases HistNetQ and GMNet, two neural quantifiers that -- unlike the
# classify-then-aggregate methods seen in most other examples (e.g., EMQ, PACC) -- are trained
# end-to-end on whole samples ("bags") of known prevalence, instead of on individually labelled
# instances. Both require `torch` to be installed (GMNet additionally requires `geotorch`); see
# quapy.method._neural_bags.BagTrainedQuantifier, the base class they share, and the "Methods" manual
# for further details.
#
# In particular, we exercise `fit_from_samples`, the entry point meant for the situation in which
# individually labelled training instances are NOT available at all, and all one has access to is a
# pool of pre-built bags of known prevalence -- e.g., as released for the LeQua challenges, or as
# collected by a third party. We simulate such a pool here using CIFAR10 (see also example 20), by
# drawing a fixed set of bags from its (otherwise fully instance-labelled) training collection with
# the UPP protocol, and then training HistNetQ and GMNet as if that pool of bags -- and not the
# underlying instance-level labels -- were all we had access to.
#
# The settings below (number of bags, training epochs, patience) are deliberately modest so that the
# example runs in a reasonable time on a CPU; scale them up for serious experimentation.

if __name__ == '__main__':

    qp.environ['SAMPLE_SIZE'] = 500

    # The datasets available in quapy do not consist of raw image files, but are instead
    # pre-generated embeddings (see the manuals for further information); 'features' corresponds to
    # the penultimate-layer representations of a resnet18, which is what HistNetQ and GMNet expect to
    # receive, since both operate on continuous instance representations.
    print('fetching cifar10 embeddings')
    train, test = fetch_image_embeddings(dataset_name='cifar10', embedding='features').train_test
    print('training:', train)
    print('test:', test)

    # Let us pretend that, instead of having access to the individually labelled instances of `train`,
    # we are only given a pool of 1000 pre-built bags of 500 instances each, with heterogeneous class
    # prevalence values (sampled uniformly at random from the simplex). This is exactly what
    # fit_from_samples expects: an AbstractProtocol that already yields (sample, prevalence) pairs.
    # UPP is deterministic by default (fixed random_state=0), so this pool is the same every time it
    # is iterated, as befits a fixed, pre-built collection of bags.
    print('simulating a pool of 1000 pre-built training bags (as if released by a third party)')
    bag_pool = UPP(train, repeats=1000, random_state=0)

    # Neither method requires a classifier, nor (in this case) a feature extraction module: since the
    # CIFAR10 embeddings are already vectorized, the default identity module is used, and both
    # networks operate directly on the 512-dimensional resnet18 features
    models = {
        'HistNetQ': HistNetQ(n_bins=8, bag_size=qp.environ['SAMPLE_SIZE'], train_epochs=50, patience=10, verbose=1),
        'GMNet': GMNet(n_gm_layers=8, num_gaussians=10, gaussian_dimensions=5, hidden_size_fe=(50,),
                        bag_size=qp.environ['SAMPLE_SIZE'], train_epochs=50, patience=10, verbose=1),
    }

    for name, model in models.items():
        print(f'\nfitting {name} from the pre-built bag pool via fit_from_samples '
              f'(no instance-level labels are used)')
        model.fit_from_samples(bag_pool)

    # we now evaluate both quantifiers on cifar10's test set, using a fresh artificial-prevalence
    # protocol (note that, at prediction time, HistNetQ and GMNet behave just like any other
    # quantifier: the bag-based training is an internal detail of these two particular methods)
    test_prot = UPP(test, repeats=1000)

    for name, model in models.items():
        report = qp.evaluation.evaluation_report(model, protocol=test_prot, error_metrics=['mae'])
        print(f'\n{name}:')
        print(report.mean(numeric_only=True))
