import unittest

from tra_bo.evaluation import (
    incumbent_future_loss_trajectory,
    normalized_final_regret,
    normalized_future_regret_auc,
)
from tra_bo.splits import make_ordered_split


class EvaluationProtocolTest(unittest.TestCase):
    def test_four_validation_windows_precede_future_window(self) -> None:
        split = make_ordered_split(length=2000, horizon=96, validation_windows=4)
        self.assertEqual(len(split.validation), 4)
        windows = (*split.validation, split.future)
        for previous, current in zip(windows, windows[1:]):
            self.assertEqual(previous.target_end, current.target_start)
        for window in windows:
            self.assertEqual(window.train_end, window.target_start)

    def test_future_regret_metrics(self) -> None:
        pool = [1.0, 2.0, 3.0]
        trajectory = incumbent_future_loss_trajectory([2, 1, 0], pool)
        self.assertEqual(trajectory.tolist(), [3.0, 2.0, 1.0])
        self.assertAlmostEqual(normalized_future_regret_auc(trajectory, pool), 0.5)
        self.assertAlmostEqual(normalized_final_regret(2.0, pool), 0.5)


if __name__ == "__main__":
    unittest.main()
