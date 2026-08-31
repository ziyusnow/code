import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


CASE2_DIR = Path(__file__).resolve().parents[1]
if str(CASE2_DIR) not in sys.path:
    sys.path.insert(0, str(CASE2_DIR))

import oracle_action_branch as oracle
import solve_case2 as solver
from experiment_types import Action, ActionParameters, SearchConfig


class OracleBranchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.p_vital, cls.p_nonvital, cls.p_pv = solver.load_input_data(
            CASE2_DIR.parent / "data.md"
        )

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="oracle-test-", dir=str(CASE2_DIR)))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_select_seeds_uses_cost_quantile_ranks(self):
        root = self.temp_dir / "uniform_random_formal" / "UniformRandom"
        for seed in range(30):
            directory = root / ("seed_%d" % seed)
            directory.mkdir(parents=True)
            (directory / "summary.json").write_text(
                json.dumps({"status": "complete", "seed": seed, "best_cost": float(seed)}),
                encoding="utf-8",
            )
        selected = oracle.select_seeds(self.temp_dir / "uniform_random_formal", 5)
        self.assertEqual([item["seed"] for item in selected], [0, 7, 14, 22, 29])

    def test_checkpoint_round_trip_preserves_state_and_rng(self):
        config = SearchConfig(population_size=8, max_iterations=2, decision_interval=1)
        evaluator = lambda positions: solver.evaluate(
            positions, self.p_vital, self.p_nonvital, self.p_pv,
            penalty_lambda=config.penalty_lambda,
        )
        streams = solver.make_random_streams(19)
        state = solver.initialize_search(evaluator, streams.core, config)
        checkpoint = {"fraction": 0.2, "target_nfe": 1, "label": "20pct", "actual_nfe": state.search_nfe, "actual_fraction": state.search_nfe / config.search_budget, "iteration": state.iteration}
        directory = self.temp_dir / "checkpoint"
        oracle.save_checkpoint(directory, state, streams, 19, checkpoint)
        restored, restored_streams, metadata = oracle.load_checkpoint(directory)
        self.assertEqual(restored.iteration, state.iteration)
        self.assertEqual(restored.search_nfe, state.search_nfe)
        self.assertEqual(restored.global_best_cost, state.global_best_cost)
        np.testing.assert_array_equal(restored.positions, state.positions)
        np.testing.assert_array_equal(restored.evaluation["cv"], state.evaluation["cv"])
        self.assertEqual(
            restored_streams.core.bit_generator.state,
            streams.core.bit_generator.state,
        )
        self.assertEqual(metadata["checkpoint"]["label"], "20pct")

    def test_action_window_has_exact_nfe_and_does_not_mutate_checkpoint(self):
        config = SearchConfig(population_size=8, max_iterations=3, decision_interval=1)
        evaluator = lambda positions: solver.evaluate(
            positions, self.p_vital, self.p_nonvital, self.p_pv,
            penalty_lambda=config.penalty_lambda,
        )
        streams = solver.make_random_streams(23)
        state = solver.initialize_search(evaluator, streams.core, config)
        directory = self.temp_dir / "checkpoint"
        checkpoint = {"fraction": 0.2, "target_nfe": 1, "label": "20pct", "actual_nfe": state.search_nfe, "actual_fraction": state.search_nfe / config.search_budget, "iteration": state.iteration}
        oracle.save_checkpoint(directory, state, streams, 23, checkpoint)
        original, original_streams, _ = oracle.load_checkpoint(directory)
        branch_state, _, _ = oracle.load_checkpoint(directory)
        branch_streams = oracle._branch_streams(original_streams, 23, original.iteration, 1, Action.A2)
        result = oracle.run_action_window(
            branch_state,
            branch_streams,
            Action.A2,
            self.p_vital,
            self.p_nonvital,
            self.p_pv,
            config,
            ActionParameters(),
            iterations=2,
        )
        self.assertEqual(result["end_nfe"], original.search_nfe + 2 * config.population_size)
        self.assertEqual(result["end_iteration"], original.iteration + 2)
        np.testing.assert_array_equal(original.positions, state.positions)
        self.assertEqual(original.search_nfe, state.search_nfe)

    def test_action_ranking_is_descending_with_deterministic_ties(self):
        ordered, text = oracle._rank_actions({Action.A1: 1.0, Action.A2: 1.0, Action.A3: 2.0, Action.A4: -1.0})
        self.assertEqual(ordered, [Action.A3, Action.A1, Action.A2, Action.A4])
        self.assertEqual(text, "A3>A1>A2>A4")

    def test_oracle_metrics_compare_best_with_uniform_action_mean(self):
        rewards = {
            Action.A1: 1.0,
            Action.A2: 3.0,
            Action.A3: 5.0,
            Action.A4: 7.0,
        }
        stds = {action: 0.5 for action in Action}
        gap_rewards = {action: reward / 100.0 for action, reward in rewards.items()}
        result = oracle._oracle_metrics(rewards, stds, gap_rewards)
        self.assertEqual(result["oracle_action"], "A4")
        self.assertEqual(result["second_best_action"], "A3")
        self.assertAlmostEqual(result["oracle_reward"], 7.0)
        self.assertAlmostEqual(result["random_reward_mean"], 4.0)
        self.assertAlmostEqual(result["delta_reward"], 3.0)
        self.assertAlmostEqual(result["relative_delta_vs_random"], 0.75)
        self.assertAlmostEqual(result["margin"], 2.0)
        self.assertAlmostEqual(result["pooled_repeat_std"], 0.5)
        self.assertAlmostEqual(result["delta_gap_reward"], 0.03)
        self.assertEqual(result["oracle_class"], "Clear Oracle")

    def test_clear_oracle_threshold_is_strict_and_ties_are_ambiguous(self):
        gap_rewards = {action: 0.0 for action in Action}
        exact_threshold = oracle._oracle_metrics(
            {Action.A1: 3.0, Action.A2: 1.0, Action.A3: 0.0, Action.A4: -1.0},
            {action: 1.0 for action in Action},
            gap_rewards,
        )
        self.assertEqual(exact_threshold["margin"], 2.0)
        self.assertEqual(exact_threshold["oracle_class"], "Ambiguous Oracle")

        tied = oracle._oracle_metrics(
            {Action.A1: 5.0, Action.A2: 5.0, Action.A3: 1.0, Action.A4: 0.0},
            {action: 0.0 for action in Action},
            gap_rewards,
        )
        self.assertEqual(tied["oracle_action"], "A1")
        self.assertEqual(tied["second_best_action"], "A2")
        self.assertEqual(tied["margin"], 0.0)
        self.assertEqual(tied["oracle_class"], "Ambiguous Oracle")


if __name__ == "__main__":
    unittest.main()
