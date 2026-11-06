import json
import unittest

from tra_bo.llm import BOState, LLMAcquisitionSelector


class RecordedResponseClient:
    model = "test-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return self.response


def state() -> BOState:
    return BOState(
        tra=0.35,
        tra_valid=True,
        evaluations=16,
        initial_evaluations=16,
        remaining_budget=34,
        total_budget=50,
        dimension=6,
        incumbent=0.2,
        incumbent_history=(0.5, 0.3, 0.2),
        target_mean=0.4,
        target_std=0.1,
        target_range=0.5,
        latest_min_distance=0.3,
        gp_uncertainty_mean=0.12,
        gp_uncertainty_max=0.25,
        gp_lengthscale_min=0.2,
        gp_lengthscale_mean=0.5,
        gp_lengthscale_max=0.8,
        gp_output_scale=1.0,
        previous_acquisition=None,
        acquisition_history=(),
        improvement_history=(),
    )


class LLMSelectorTest(unittest.TestCase):
    def test_valid_json_selects_acquisition(self) -> None:
        client = RecordedResponseClient('{"acquisition": "UCB"}')
        selector = LLMAcquisitionSelector(client=client)
        self.assertEqual(selector.select(state()), "UCB")
        payload = json.loads(client.messages[-1]["content"])
        self.assertEqual(payload["state"]["tra"], 0.35)
        self.assertEqual(payload["state"]["incumbent_history"], [0.5, 0.3, 0.2])
        self.assertEqual(payload["state"]["gp_uncertainty_max"], 0.25)
        self.assertEqual(payload["selection_protocol"], "select_acquisition_then_optimize")

    def test_invalid_response_uses_ei_fallback(self) -> None:
        selector = LLMAcquisitionSelector(client=RecordedResponseClient("invalid"))
        self.assertEqual(selector.select(state()), "EI")
        self.assertTrue(selector.calls[-1]["fallback_used"])


if __name__ == "__main__":
    unittest.main()
