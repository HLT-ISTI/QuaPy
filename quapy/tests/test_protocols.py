import unittest
import numpy as np

import quapy.functional
from quapy.data import LabelledCollection
from quapy.protocol import APP, NPP, UPP, DomainMixer, AbstractStochasticSeededProtocol


def mock_labelled_collection(prefix=''):
    y = [0] * 250 + [1] * 250 + [2] * 250 + [3] * 250
    X = [prefix + str(i) + '-' + str(yi) for i, yi in enumerate(y)]
    return LabelledCollection(X, y, classes=sorted(np.unique(y)))


def samples_to_str(protocol):
    samples_str = ""
    for instances, prev in protocol():
        samples_str += f'{instances}\t{prev}\n'
    return samples_str


class TestProtocols(unittest.TestCase):

    def test_app_sanity_check(self):
        data = mock_labelled_collection()
        n_prevpoints = 101
        repeats = 10
        with self.assertRaises(RuntimeError):
            p = APP(data, sample_size=5, n_prevalences=n_prevpoints, repeats=repeats, random_state=42)
        n_combinations = \
            quapy.functional.num_prevalence_combinations(n_prevpoints, n_classes=data.n_classes, n_repeats=repeats)
        p = APP(data, sample_size=5, n_prevalences=n_prevpoints, random_state=42, sanity_check=n_combinations)
        p = APP(data, sample_size=5, n_prevalences=n_prevpoints, random_state=42, sanity_check=None)

    def test_app_replicate(self):
        data = mock_labelled_collection()
        p = APP(data, sample_size=5, n_prevalences=11, random_state=42)

        samples1 = samples_to_str(p)
        samples2 = samples_to_str(p)

        self.assertEqual(samples1, samples2)

        p = APP(data, sample_size=5, n_prevalences=11)  # <- random_state is by default set to 0

        samples1 = samples_to_str(p)
        samples2 = samples_to_str(p)

        self.assertEqual(samples1, samples2)

    def test_app_not_replicate(self):
        data = mock_labelled_collection()
        p = APP(data, sample_size=5, n_prevalences=11, random_state=None)

        samples1 = samples_to_str(p)
        samples2 = samples_to_str(p)

        self.assertNotEqual(samples1, samples2)

        p = APP(data, sample_size=5, n_prevalences=11, random_state=42)
        samples1 = samples_to_str(p)
        p = APP(data, sample_size=5, n_prevalences=11, random_state=0)
        samples2 = samples_to_str(p)

        self.assertNotEqual(samples1, samples2)

    def test_app_number(self):
        data = mock_labelled_collection()
        p = APP(data, sample_size=100, n_prevalences=10, repeats=1)

        # surprisingly enough, for some n_prevalences the test fails, notwithstanding
        # everything is correct. The problem is that in function APP.prevalence_grid()
        # there is sometimes one rounding error that gets cumulated and
        # surpasses 1.0 (by a very small float value, 0.0000000000002 or sthe like)
        # so these tuples are mistakenly removed... I have tried with np.close, and
        # other workarounds, but eventually happens that there is some negative probability
        # in the sampling function...

        count = 0
        for _ in p():
            count+=1

        self.assertEqual(count, p.total())

    def test_npp_replicate(self):
        data = mock_labelled_collection()
        p = NPP(data, sample_size=5, repeats=5, random_state=42)

        samples1 = samples_to_str(p)
        samples2 = samples_to_str(p)

        self.assertEqual(samples1, samples2)

        p = NPP(data, sample_size=5, repeats=5)  # <- random_state is by default set to 0

        samples1 = samples_to_str(p)
        samples2 = samples_to_str(p)

        self.assertEqual(samples1, samples2)

    def test_npp_not_replicate(self):
        data = mock_labelled_collection()
        p = NPP(data, sample_size=5, repeats=5, random_state=None)

        samples1 = samples_to_str(p)
        samples2 = samples_to_str(p)

        self.assertNotEqual(samples1, samples2)

        p = NPP(data, sample_size=5, repeats=5, random_state=42)
        samples1 = samples_to_str(p)
        p = NPP(data, sample_size=5, repeats=5, random_state=0)
        samples2 = samples_to_str(p)
        self.assertNotEqual(samples1, samples2)

    def test_kraemer_replicate(self):
        data = mock_labelled_collection()
        p = UPP(data, sample_size=5, repeats=10, random_state=42)

        samples1 = samples_to_str(p)
        samples2 = samples_to_str(p)

        self.assertEqual(samples1, samples2)

        p = UPP(data, sample_size=5, repeats=10)  # <- random_state is by default set to 0

        samples1 = samples_to_str(p)
        samples2 = samples_to_str(p)

        self.assertEqual(samples1, samples2)

    def test_kraemer_not_replicate(self):
        data = mock_labelled_collection()
        p = UPP(data, sample_size=5, repeats=10, random_state=None)

        samples1 = samples_to_str(p)
        samples2 = samples_to_str(p)

        self.assertNotEqual(samples1, samples2)

    def test_covariate_shift_replicate(self):
        dataA = mock_labelled_collection('domA')
        dataB = mock_labelled_collection('domB')
        p = DomainMixer(dataA, dataB, sample_size=10, mixture_points=11, random_state=1)

        samples1 = samples_to_str(p)
        samples2 = samples_to_str(p)

        self.assertEqual(samples1, samples2)

        p = DomainMixer(dataA, dataB, sample_size=10, mixture_points=11)  # <- random_state is by default set to 0

        samples1 = samples_to_str(p)
        samples2 = samples_to_str(p)

        self.assertEqual(samples1, samples2)

    def test_covariate_shift_not_replicate(self):
        dataA = mock_labelled_collection('domA')
        dataB = mock_labelled_collection('domB')
        p = DomainMixer(dataA, dataB, sample_size=10, mixture_points=11, random_state=None)

        samples1 = samples_to_str(p)
        samples2 = samples_to_str(p)

        self.assertNotEqual(samples1, samples2)

    def test_no_seed_init(self):
        class NoSeedInit(AbstractStochasticSeededProtocol):
            def __init__(self):
                self.data = mock_labelled_collection()

            def samples_parameters(self):
                # return a matrix containing sampling indexes in the rows
                return np.random.randint(0, len(self.data), 10*10).reshape(10, 10)

            def sample(self, params):
                index = np.unique(params)
                return self.data.sampling_from_index(index)

        p = NoSeedInit()

        # this should raise a ValueError, since the class is said to be AbstractStochasticSeededProtocol but the
        # random_seed has never been passed to super(NoSeedInit, self).__init__(random_seed)
        with self.assertRaises(ValueError):
            for sample in p():
                pass
            print('done')


if __name__ == '__main__':
    unittest.main()
