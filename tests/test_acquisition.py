import unittest

import numpy as np

from tra_bo.acquisition import ACQUISITIONS, acquisition_scores
from tra_bo.surrogate import MaternGaussianProcess


class AcquisitionTest(unittest.TestCase):
    def test_all_acquisitions_score_the_remaining_pool(self) -> None:
        observed_x = np.asarray([[0.0], [0.35], [0.7], [1.0]])
        observed_y = np.asarray([0.8, 0.2, 0.4, 0.9])
        candidates = np.asarray([[0.15], [0.5], [0.85]])
        surrogate = MaternGaussianProcess(
            dimension=1, random_state=3, n_restarts=0
        ).fit(observed_x, observed_y)

        for action in ACQUISITIONS:
            scores = acquisition_scores(
                action,
                surrogate,
                candidates,
                observed_y,
                np.random.default_rng(9),
            )
            self.assertEqual(scores.shape, (len(candidates),))
            self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
