import unittest

import numpy as np

from tra_bo.llm import FixedAcquisitionSelector, LLMAcquisitionSelector
from tra_bo.optimizer import TRABO, TRABOConfig
from tra_bo.temporal import temporal_target


class OptimizerTest(unittest.TestCase):
    def test_precomputed_visible_window_losses(self) -> None:
        rng = np.random.default_rng(11)
        candidates = rng.uniform(size=(12, 2))
        losses = np.column_stack(
            [
                np.square(candidates - center).sum(axis=1) + 0.1
                for center in (0.2, 0.3, 0.5, 0.6)
            ]
        )
        result = TRABO(
            TRABOConfig(
                budget=8,
                n_initial=4,
                block_size=2,
                gp_restarts=0,
                random_state=4,
            ),
            FixedAcquisitionSelector("EI"),
        ).optimize_precomputed(candidates, losses)
        self.assertEqual(len(result.observed_indices), 8)
        for record in result.trials:
            np.testing.assert_allclose(
                record.window_losses, losses[record.candidate_index]
            )

    def test_dynamic_tra_updates_at_block_boundaries_and_budget(self) -> None:
        rng = np.random.default_rng(7)
        candidates = rng.uniform(size=(32, 2))

        def objective(x: np.ndarray) -> list[float]:
            early = float(np.square(x - np.asarray([0.2, 0.8])).sum() + 0.1)
            recent = float(np.square(x - np.asarray([0.5, 0.5])).sum() + 0.1)
            return [early, 1.05 * early, 0.8 * early + 0.2 * recent, recent]

        config = TRABOConfig(
            budget=13,
            n_initial=6,
            block_size=3,
            gp_restarts=0,
            random_state=3,
        )
        result = TRABO(config, FixedAcquisitionSelector("EI")).optimize(
            candidates, objective
        )
        self.assertEqual(len(result.observed_indices), 13)
        self.assertEqual(
            len(result.incumbent_indices), config.budget - config.n_initial + 1
        )
        self.assertTrue(
            set(result.incumbent_indices).issubset(result.observed_indices)
        )
        self.assertEqual([step for step, _ in result.tra_history], [6, 9, 12, 13])
        self.assertEqual(result.trials[6].decision_tra, result.tra_history[0][1])
        self.assertIn(result.best_index, result.observed_indices)
        self.assertEqual(len(set(result.observed_indices)), 13)
        observed_losses = np.asarray(
            [objective(candidates[index]) for index in result.observed_indices]
        )
        expected_targets = temporal_target(
            observed_losses,
            recent_count=config.recent_count,
            q=result.final_tra,
            eta=config.eta,
        )
        np.testing.assert_allclose(result.final_targets, expected_targets)

    def test_llm_selector_runs_inside_the_full_loop(self) -> None:
        class AlwaysEIClient:
            model = "test-model"

            def complete(self, messages):
                self.last_messages = messages
                return '{"acquisition": "EI"}'

        rng = np.random.default_rng(5)
        candidates = rng.uniform(size=(16, 2))

        def objective(x: np.ndarray) -> list[float]:
            value = float(np.square(x - 0.4).sum() + 0.1)
            return [1.1 * value, value, 0.95 * value, 0.9 * value]

        selector = LLMAcquisitionSelector(client=AlwaysEIClient())
        result = TRABO(
            TRABOConfig(
                budget=8,
                n_initial=4,
                block_size=2,
                gp_restarts=0,
                random_state=2,
            ),
            selector,
        ).optimize(candidates, objective)
        self.assertEqual(len(result.llm_calls), 4)
        self.assertTrue(all(call["acquisition"] == "EI" for call in result.llm_calls))
        self.assertEqual(len(result.observed_indices), 8)


if __name__ == "__main__":
    unittest.main()
