from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


CASE2_DIR = Path(__file__).resolve().parents[1]
if str(CASE2_DIR) not in sys.path:
    sys.path.insert(0, str(CASE2_DIR))

from experiment_types import (
    Action,
    Phase,
    SelectionDecision,
    SelectorErrorKind,
    SelectorSummary,
    StrategyOutcome,
)
from strategy_selectors import (
    ManualSelector,
    ReplaySelector,
    RuleSelector,
    UCB1Selector,
    UniformRandomSelector,
    create_selector,
)


def make_summary(window=0, phase=Phase.FEASIBILITY, **overrides):
    summary = SelectorSummary(
        window_index=window,
        iteration=window * 20,
        search_nfe=600 + window * 12000,
        budget_fraction=(600 + window * 12000) / 300600.0,
        phase=phase,
        global_best_cost=43000.0,
        global_best_cv=2.0 if phase == Phase.FEASIBILITY else 0.0,
        cv_r=1.0,
        cv_soc=0.5,
        cv_e=0.3,
        cv_d=0.2,
        feasible_fraction=0.0 if phase == Phase.FEASIBILITY else 0.1,
        population_cv_q25=1.0,
        population_cv_median=2.0,
        population_cv_q75=3.0,
        normalized_diversity=0.2,
        stagnation_iterations=0,
        previous_action=None,
    )
    return replace(summary, **overrides)


def make_outcome(action, window=0, phase=Phase.FEASIBILITY, reward=0.25, **overrides):
    outcome = StrategyOutcome(
        window_index=window,
        phase_at_start=phase,
        phase_at_end=phase,
        start_iteration=window * 20,
        end_iteration=(window + 1) * 20,
        start_nfe=600 + window * 12000,
        end_nfe=600 + (window + 1) * 12000,
        start_best_cost=43000.0,
        end_best_cost=42900.0,
        start_best_cv=2.0,
        end_best_cv=1.5,
        first_feasible_appeared=False,
        first_feasible_nfe=None,
        relative_improvement=reward,
        requested_action=action,
        applied_action=action,
        fallback_used=False,
        error_kind=None,
        elapsed_seconds=0.0,
        llm_call_id=None,
    )
    return replace(outcome, **overrides)


class SelectorTests(unittest.TestCase):
    def test_manual_fixed_and_exact_sequence(self):
        rng = np.random.default_rng(1)
        fixed = ManualSelector(action=Action.A3)
        self.assertEqual(fixed.select(make_summary(), rng).applied_action, Action.A3)

        sequence = [Action.A1, Action.A2, Action.A3]
        manual = ManualSelector(sequence=sequence, expected_windows=3)
        selected = [
            manual.select(make_summary(window=index), rng).applied_action
            for index in range(3)
        ]
        self.assertEqual(selected, sequence)
        with self.assertRaises(ValueError):
            ManualSelector(sequence=sequence, expected_windows=25)
        with self.assertRaises(ValueError):
            ManualSelector(action=Action.A1, sequence=sequence)

    def test_uniform_random_reproduces_without_touching_core_rng(self):
        selector_a = UniformRandomSelector()
        selector_b = UniformRandomSelector()
        rng_a = np.random.default_rng(22)
        rng_b = np.random.default_rng(22)
        actions_a = [selector_a.select(make_summary(i), rng_a).applied_action for i in range(25)]
        actions_b = [selector_b.select(make_summary(i), rng_b).applied_action for i in range(25)]
        self.assertEqual(actions_a, actions_b)
        self.assertTrue(all(action in Action for action in actions_a))

        core_a = np.random.default_rng(91)
        core_b = np.random.default_rng(91)
        selector_rng = np.random.default_rng(92)
        for index in range(100):
            selector_a.select(make_summary(index), selector_rng)
        np.testing.assert_array_equal(core_a.random(100), core_b.random(100))

    def test_rule_priority_and_boundaries(self):
        selector = RuleSelector()
        rng = np.random.default_rng(1)
        self.assertEqual(
            selector.select(
                make_summary(phase=Phase.FEASIBILITY, budget_fraction=0.9), rng
            ).applied_action,
            Action.A4,
        )
        self.assertEqual(
            selector.select(
                make_summary(phase=Phase.COST, budget_fraction=0.8), rng
            ).applied_action,
            Action.A3,
        )
        self.assertEqual(
            selector.select(
                make_summary(
                    phase=Phase.COST,
                    budget_fraction=0.79,
                    stagnation_iterations=20,
                ),
                rng,
            ).applied_action,
            Action.A2,
        )
        self.assertEqual(
            selector.select(make_summary(phase=Phase.COST), rng).applied_action,
            Action.A1,
        )

    def test_ucb_forced_order_and_observe_lifecycle(self):
        selector = UCB1Selector()
        rng = np.random.default_rng(4)
        actions = []
        for window, expected in enumerate(Action):
            decision = selector.select(make_summary(window), rng)
            actions.append(decision.applied_action)
            self.assertEqual(decision.applied_action, expected)
            selector.observe(
                expected,
                make_outcome(expected, window=window, reward=0.1 * (window + 1)),
            )
        self.assertEqual(actions, list(Action))
        with self.assertRaises(RuntimeError):
            selector.observe(Action.A1, make_outcome(Action.A1))

    def test_ucb_tie_order_reward_clip_and_phase_reset(self):
        selector = UCB1Selector()
        rng = np.random.default_rng(4)
        for window, action in enumerate(Action):
            selector.select(make_summary(window), rng)
            selector.observe(action, make_outcome(action, window=window, reward=5.0))
        next_action = selector.select(make_summary(4), rng).applied_action
        self.assertEqual(next_action, Action.A1)
        selector.observe(Action.A1, make_outcome(Action.A1, window=4, reward=-5.0))
        self.assertEqual(selector.statistics()["mean_rewards"]["A1"], 0.0)

        cost_decision = selector.select(
            make_summary(5, phase=Phase.COST), rng
        )
        self.assertEqual(cost_decision.applied_action, Action.A1)
        self.assertIn(Phase.FEASIBILITY, selector.archives)
        self.assertEqual(selector.statistics()["counts"]["A1"], 0)
        selector.observe(
            Action.A1,
            make_outcome(Action.A1, window=5, phase=Phase.COST),
        )
        with self.assertRaises(ValueError):
            selector.select(make_summary(6, phase=Phase.FEASIBILITY), rng)

    def test_ucb_transition_window_reward_stays_in_feasibility_archive(self):
        selector = UCB1Selector()
        rng = np.random.default_rng(5)
        action = selector.select(make_summary(), rng).applied_action
        selector.observe(
            action,
            make_outcome(
                action,
                phase=Phase.FEASIBILITY,
                phase_at_end=Phase.COST,
                first_feasible_appeared=True,
                first_feasible_nfe=12600,
                reward=1.0,
            ),
        )
        selector.select(make_summary(1, phase=Phase.COST), rng)
        archived = selector.archives[Phase.FEASIBILITY]
        self.assertEqual(archived["counts"]["A1"], 1)
        self.assertEqual(archived["mean_rewards"]["A1"], 1.0)

    def test_replay_validates_metadata_and_does_not_consume_rng(self):
        records = [
            {
                "window_index": index,
                "start_iteration": index * 20,
                "end_iteration": (index + 1) * 20,
                "start_nfe": 600 + index * 12000,
                "end_nfe": 600 + (index + 1) * 12000,
                "phase_at_start": Phase.FEASIBILITY.value,
                "phase_at_end": Phase.FEASIBILITY.value,
                "start_best_cost": 43000.0,
                "end_best_cost": 42900.0,
                "start_best_cv": 2.0,
                "end_best_cv": 1.5,
                "first_feasible_appeared": False,
                "first_feasible_nfe": None,
                "relative_improvement": 0.25,
                "requested_action": Action.A2.value,
                "applied_action": Action.A2.value,
                "fallback_used": False,
                "error_kind": None,
                "llm_call_id": None,
            }
            for index in range(3)
        ]
        replay = ReplaySelector(records, expected_windows=3)
        rng = np.random.default_rng(8)
        control = np.random.default_rng(8)
        for index in range(3):
            decision = replay.select(make_summary(index), rng)
            self.assertEqual(decision.applied_action, Action.A2)
            replay.observe(Action.A2, make_outcome(Action.A2, window=index))
        np.testing.assert_array_equal(rng.random(10), control.random(10))

        bad = list(records)
        bad[1] = dict(bad[1], start_nfe=999)
        replay = ReplaySelector(bad, expected_windows=3)
        replay.select(make_summary(0), np.random.default_rng(1))
        with self.assertRaises(ValueError):
            replay.select(make_summary(1), np.random.default_rng(1))

        bad_outcome = list(records)
        bad_outcome[0] = dict(bad_outcome[0], end_nfe=999)
        replay = ReplaySelector(bad_outcome, expected_windows=3)
        replay.select(make_summary(0), np.random.default_rng(1))
        with self.assertRaises(ValueError):
            replay.observe(Action.A2, make_outcome(Action.A2))

    def test_create_selector_builds_local_methods_and_rejects_unknown(self):
        for method, expected_type in (
            ("A1-only", ManualSelector),
            ("UniformRandom", UniformRandomSelector),
            ("Rule", RuleSelector),
            ("UCB1", UCB1Selector),
        ):
            spec = SimpleNamespace(method=method)
            self.assertIsInstance(create_selector(spec), expected_type)
        with self.assertRaises(ValueError):
            create_selector(SimpleNamespace(method="unknown"))


if __name__ == "__main__":
    unittest.main()
