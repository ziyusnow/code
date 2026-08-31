import copy
import os
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np


CASE2_DIR = Path(__file__).resolve().parents[1]
if str(CASE2_DIR) not in sys.path:
    sys.path.insert(0, str(CASE2_DIR))

import solve_case2 as solver
from experiment_types import Action, ActionParameters, SelectionDecision


class RecordingSelector:
    def __init__(self, action=Action.A1):
        self.action = action
        self.summaries = []
        self.observations = []

    def select(self, summary, rng):
        self.summaries.append(summary)
        return SelectionDecision(
            requested_action=self.action,
            applied_action=self.action,
            elapsed_seconds=0.125,
            llm_call_id="call-test",
        )

    def observe(self, action, outcome):
        self.observations.append((action, outcome))


class RaisingSelector:
    def select(self, summary, rng):
        raise RuntimeError("selector failed")

    def observe(self, action, outcome):
        pass


class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data_path = CASE2_DIR.parent / "data.md"
        _, cls.p_vital, cls.p_nonvital, cls.p_pv = solver.load_input_data(
            data_path
        )

    def small_config(self, population_size=8, max_iterations=3):
        return replace(
            solver.default_search_config(),
            population_size=population_size,
            max_iterations=max_iterations,
            decision_interval=2,
        )

    def make_state(self, config=None, seed=11):
        config = config or self.small_config()
        evaluator = lambda positions: solver.evaluate(
            positions, self.p_vital, self.p_nonvital, self.p_pv
        )
        return solver.initialize_search(
            evaluator, np.random.default_rng(seed), config
        )

    def test_canonicalize_rejects_shape_and_nonfinite_values(self):
        config = self.small_config()
        with self.assertRaises(ValueError):
            solver.canonicalize(np.zeros((config.population_size, 71)), config)
        with self.assertRaises(ValueError):
            solver.canonicalize(np.zeros(72), config)
        values = np.zeros((config.population_size, 72))
        values[0, 0] = np.nan
        with self.assertRaises(ValueError):
            solver.canonicalize(values, config)
        values[0, 0] = np.inf
        with self.assertRaises(ValueError):
            solver.canonicalize(values, config)

    def test_canonicalize_projects_without_mutating_input(self):
        config = self.small_config()
        rng = np.random.default_rng(3)
        values = rng.uniform(-5.0, 15.0, (config.population_size, 72))
        original = values.copy()
        canonical = solver.canonicalize(values, config)
        np.testing.assert_array_equal(values, original)
        self.assertTrue(np.isfinite(canonical).all())
        self.assertTrue(np.all(canonical[:, :24] >= 0.0))
        self.assertTrue(np.all(canonical[:, :24] <= 11.0))
        np.testing.assert_allclose(canonical[:, :24].sum(axis=1), 240.0)
        self.assertTrue(np.all(canonical[:, 24:] >= 0.0))
        self.assertTrue(np.all(canonical[:, 24:] <= 1.0))

    def test_rank_indices_uses_feasibility_first_and_stable_ties(self):
        cost = np.array([100.0, 80.0, 40.0, 40.0, 10.0])
        cv = np.array([0.0, 1.0, 0.0, 0.0, 0.5])
        order = solver.rank_indices(cost, cv)
        np.testing.assert_array_equal(order, np.array([2, 3, 0, 4, 1]))

    def test_a1_is_identity_and_consumes_no_rng(self):
        config = self.small_config()
        state = self.make_state(config)
        rng = np.random.default_rng(4)
        before = copy.deepcopy(rng.bit_generator.state)
        result = solver.strategy_a1(state.positions, state, rng)
        self.assertIs(result, state.positions)
        self.assertEqual(before, rng.bit_generator.state)

    def test_a2_donors_are_distinct_and_exclude_target(self):
        targets = np.array([0, 2, 5, 7])
        donors = solver._sample_a2_donors(
            targets, 8, np.random.default_rng(5)
        )
        self.assertEqual(donors.shape, (4, 3))
        for target, row in zip(targets, donors):
            self.assertEqual(len(set(row.tolist())), 3)
            self.assertNotIn(target, row)

    def test_a2_forces_a_donor_dimension_when_cr_is_zero(self):
        config = self.small_config()
        state = self.make_state(config)
        x_base = np.arange(config.population_size * 72, dtype=float).reshape(
            config.population_size, 72
        )
        parameters = ActionParameters(a2_crossover_probability=0.0)
        transformed = solver.strategy_a2(
            x_base,
            state,
            np.random.default_rng(6),
            parameters,
            config,
        )
        order = solver.rank_indices(
            state.evaluation["total_cost"], state.evaluation["cv"]
        )
        targets = order[-int(config.population_size * 0.25) :]
        self.assertTrue(np.all(np.any(transformed[targets] != x_base[targets], axis=1)))
        canonical = solver.canonicalize(transformed, config)
        np.testing.assert_allclose(canonical[:, :24].sum(axis=1), 240.0)
        self.assertTrue(np.all((canonical[:, 24:] >= 0.0) & (canonical[:, 24:] <= 1.0)))

    def test_a3_perturbs_declared_best_target_count(self):
        config = self.small_config()
        parameters = ActionParameters(a3_target_fraction=0.25)
        state = self.make_state(config)
        x_base = state.positions.copy()
        transformed = solver.strategy_a3(
            x_base,
            state,
            np.random.default_rng(7),
            parameters,
            config,
        )
        targets = solver.rank_indices(
            state.evaluation["total_cost"], state.evaluation["cv"]
        )[:2]
        untouched = np.setdiff1d(np.arange(config.population_size), targets)
        np.testing.assert_array_equal(transformed[untouched], x_base[untouched])
        self.assertTrue(np.all(np.any(transformed[targets] != x_base[targets], axis=1)))

    def test_a4_uses_minimum_cv_pbest_when_none_is_feasible(self):
        config = self.small_config(population_size=4)
        state = self.make_state(config)
        state.personal_best_cv[:] = np.array([2.0, 0.5, 1.0, 3.0])
        state.personal_best_cost[:] = np.array([1.0, 9.0, 2.0, 0.0])
        state.personal_best_positions[:] = np.arange(4 * 72).reshape(4, 72)
        state.evaluation["cv"][:] = np.array([0.2, 0.3, 0.4, 0.5])
        state.evaluation["total_cost"][:] = np.arange(4.0)
        x_base = np.zeros((4, 72))
        expected_rng = np.random.default_rng(8)
        beta = expected_rng.uniform(0.2, 0.8, (1, 1))[0, 0]
        transformed = solver.strategy_a4(
            x_base, state, np.random.default_rng(8), ActionParameters(), config
        )
        target = solver.rank_indices(
            state.evaluation["total_cost"], state.evaluation["cv"]
        )[-1]
        expected = beta * state.personal_best_positions[1]
        np.testing.assert_allclose(transformed[target], expected)

    def test_action_stream_draws_do_not_shift_core_stream(self):
        streams = solver.make_random_streams(9)
        reference = solver.make_random_streams(9)
        streams.a2.random(1000)
        streams.a3.random(1000)
        streams.a4.random(1000)
        streams.selector.random(1000)
        np.testing.assert_array_equal(streams.core.random(100), reference.core.random(100))

    def test_small_run_has_exact_nfe_and_selector_lifecycle(self):
        config = self.small_config(max_iterations=3)
        selector = RecordingSelector(Action.A2)
        run = solver.solve_experiment(
            self.p_vital,
            self.p_nonvital,
            self.p_pv,
            seed=10,
            selector=selector,
            config=config,
        )
        self.assertEqual(run.search_nfe, config.population_size * 4)
        self.assertEqual(run.audit_nfe, 1)
        self.assertEqual(len(run.history), 4)
        self.assertEqual(len(selector.summaries), 2)
        self.assertEqual(len(selector.observations), 2)
        self.assertEqual(len(run.outcomes), 2)
        self.assertEqual([summary.window_index for summary in selector.summaries], [0, 1])
        self.assertIsNone(selector.summaries[0].previous_action)
        self.assertEqual(selector.summaries[1].previous_action, Action.A2)
        self.assertEqual(run.outcomes[0].window_index, 0)
        self.assertEqual(run.outcomes[0].start_iteration, 0)
        self.assertEqual(run.outcomes[0].end_iteration, 2)
        self.assertEqual(run.outcomes[1].start_iteration, 2)
        self.assertEqual(run.outcomes[1].end_iteration, 3)
        self.assertEqual(
            run.outcomes[0].phase_at_end,
            run.outcomes[1].phase_at_start,
        )
        self.assertEqual(run.history[1]["action"], "A2")
        self.assertEqual(run.history[1]["llm_call_id"], "call-test")
        self.assertAlmostEqual(run.outcomes[0].elapsed_seconds, 0.125)

    def test_selector_exception_propagates(self):
        config = self.small_config(max_iterations=1)
        with self.assertRaisesRegex(RuntimeError, "selector failed"):
            solver.solve_experiment(
                self.p_vital,
                self.p_nonvital,
                self.p_pv,
                seed=12,
                selector=RaisingSelector(),
                config=config,
            )

    def test_search_config_controls_penalty_and_feasibility(self):
        positions = np.zeros((2, 72))
        positions[:, :24] = 10.0
        default_result = solver.evaluate(
            positions, self.p_vital, self.p_nonvital, self.p_pv
        )
        custom_result = solver.evaluate(
            positions,
            self.p_vital,
            self.p_nonvital,
            self.p_pv,
            penalty_lambda=7.0,
        )
        np.testing.assert_allclose(
            custom_result["fitness"],
            custom_result["total_cost"] + 7.0 * custom_result["cv"],
        )
        self.assertFalse(np.allclose(
            default_result["fitness"], custom_result["fitness"]
        ))

        cost = np.array([100.0, 1.0])
        cv = np.array([0.0, 0.5])
        self.assertEqual(solver.best_index(cost, cv, 0.0), 0)
        self.assertEqual(solver.best_index(cost, cv, 1.0), 1)

    def test_configuration_errors_are_rejected_before_selection(self):
        selector = RecordingSelector()
        config = replace(self.small_config(), decision_interval=0)
        with self.assertRaises(ValueError):
            solver.solve_experiment(
                self.p_vital,
                self.p_nonvital,
                self.p_pv,
                selector=selector,
                config=config,
            )
        self.assertEqual(selector.summaries, [])

        with self.assertRaises(ValueError):
            solver.solve_experiment(
                self.p_vital,
                self.p_nonvital,
                self.p_pv,
                config=self.small_config(),
                action_parameters=ActionParameters(a3_sigma_min=0.2),
            )

        with self.assertRaises(ValueError):
            solver.solve_experiment(
                self.p_vital,
                self.p_nonvital,
                self.p_pv,
                config=replace(self.small_config(), penalty_lambda=np.nan),
            )

    def test_state_advancement_keeps_mppso_velocity(self):
        config = self.small_config()
        state = self.make_state(config)
        old_positions = state.positions
        old_velocities = state.velocities
        x_base, v_base = solver.mppso_step(
            state, np.random.default_rng(13), config
        )
        positions = solver.canonicalize(x_base + 0.01, config)
        evaluation = solver.evaluate(
            positions, self.p_vital, self.p_nonvital, self.p_pv
        )
        state.search_nfe += config.population_size
        solver.advance_search_state(
            state, positions, v_base, evaluation, config
        )
        self.assertIs(state.previous_positions, old_positions)
        self.assertIs(state.previous_velocities, old_velocities)
        np.testing.assert_array_equal(state.velocities, v_base)
        np.testing.assert_array_equal(state.positions, positions)

    @unittest.skipUnless(
        os.environ.get("CASE2_FULL_REGRESSION") == "1",
        "set CASE2_FULL_REGRESSION=1 for the 500-iteration baseline",
    )
    def test_full_legacy_a1_regression(self):
        best, result, history = solver.solve(
            self.p_vital, self.p_nonvital, self.p_pv
        )
        self.assertEqual(best.shape, (72,))
        self.assertEqual(len(history), 501)
        self.assertEqual(
            set(history[0]),
            {"iteration", "best_cost", "best_cv", "best_fitness", "feasible"},
        )
        self.assertAlmostEqual(float(result["total_cost"][0]), 42433.447716, places=6)
        self.assertLessEqual(float(result["cv"][0]), 1.0e-10)
        metrics = solver.validation_metrics(
            result, self.p_vital, self.p_nonvital, self.p_pv
        )
        self.assertEqual(solver.validation_failures(result, metrics), [])


if __name__ == "__main__":
    unittest.main()
