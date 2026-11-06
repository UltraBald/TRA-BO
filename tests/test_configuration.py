import json
from pathlib import Path
import unittest

from tra_bo.llm import TRA_BO_SYSTEM_PROMPT
from tra_bo.spaces import generate_sobol_pool, space_dimension


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTest(unittest.TestCase):
    def test_default_configuration_and_benchmark_counts(self) -> None:
        protocol = json.loads((ROOT / "configs/defaults.json").read_text())
        tasks = json.loads((ROOT / "configs/benchmark_tasks.json").read_text())["tasks"]
        datasets = {task["dataset"] for task in tasks}
        forecasters = {task["forecaster"] for task in tasks}
        self.assertEqual(len(tasks), 50)
        self.assertEqual(len({task["id"] for task in tasks}), 50)
        self.assertEqual(len(datasets), 11)
        self.assertEqual(forecasters, {"dlinear", "patchtst"})
        self.assertEqual(protocol["candidate_pool"]["selectable_configurations"], 256)
        self.assertEqual(protocol["candidate_pool"]["initial_configurations"], 16)
        self.assertEqual(protocol["candidate_pool"]["query_candidates"], 240)
        self.assertEqual(protocol["candidate_pool"]["held_out_probes"], 24)
        self.assertEqual(len(protocol["optimizer_seeds"]), 5)

    def test_packaged_prompt_and_response_schema_agree(self) -> None:
        prompt_path = ROOT / "src/tra_bo/prompts/acquisition_controller.txt"
        schema = json.loads(
            (ROOT / "schemas/acquisition_response.schema.json").read_text()
        )
        self.assertEqual(TRA_BO_SYSTEM_PROMPT, prompt_path.read_text().strip())
        actions = schema["properties"]["acquisition"]["enum"]
        self.assertEqual(actions, ["PI", "EI", "UCB", "TS"])
        self.assertIn('{"acquisition": "PI|EI|UCB|TS"}', TRA_BO_SYSTEM_PROMPT)

    def test_search_spaces_generate_valid_pools(self) -> None:
        self.assertEqual(space_dimension("dlinear"), 6)
        self.assertEqual(space_dimension("patchtst"), 8)
        for forecaster in ("dlinear", "patchtst"):
            latent, configurations = generate_sobol_pool(forecaster, 256, seed=17)
            self.assertEqual(latent.shape, (256, space_dimension(forecaster)))
            self.assertEqual(len(configurations), 256)
        for config in generate_sobol_pool("patchtst", 256, seed=17)[1]:
            self.assertLessEqual(config["stride"], config["patch_len"])
            self.assertEqual(config["d_model"] % config["n_heads"], 0)


if __name__ == "__main__":
    unittest.main()
