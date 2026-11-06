import unittest

import numpy as np

from tra_bo.temporal import compute_tra, summarize_windows, temporal_target


class TemporalRankAgreementTest(unittest.TestCase):
    def test_constant_ranking_uses_neutral_fallback(self) -> None:
        losses = np.asarray(
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 3.0, 3.0],
                [1.0, 1.0, 4.0, 4.0],
            ]
        )
        estimate = compute_tra(losses, recent_count=2)
        self.assertFalse(estimate.valid)
        self.assertEqual(estimate.value, 0.5)

    def test_common_shift_preserves_rank(self) -> None:
        losses = np.asarray(
            [
                [1.0, 1.1, 3.0, 3.1],
                [2.0, 2.1, 4.0, 4.1],
                [3.0, 3.1, 5.0, 5.1],
            ]
        )
        estimate = compute_tra(losses, recent_count=2)
        self.assertTrue(estimate.valid)
        self.assertAlmostEqual(estimate.value, 1.0)

    def test_rank_reversal_produces_zero_tra(self) -> None:
        losses = np.asarray(
            [
                [1.0, 1.0, 3.0, 3.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 1.0, 1.0],
            ]
        )
        estimate = compute_tra(losses, recent_count=2)
        self.assertAlmostEqual(estimate.value, 0.0)

    def test_target_uses_more_recent_evidence_at_low_tra(self) -> None:
        losses = np.asarray([[1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 4.0, 4.0]])
        summary = summarize_windows(losses, recent_count=2)
        high = temporal_target(losses, recent_count=2, q=1.0)
        low = temporal_target(losses, recent_count=2, q=0.0, eta=0.25)
        np.testing.assert_allclose(high, summary.full_history)
        np.testing.assert_allclose(
            low, 0.75 * summary.full_history + 0.25 * summary.recent
        )


if __name__ == "__main__":
    unittest.main()
