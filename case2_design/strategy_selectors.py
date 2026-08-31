import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

try:
    from .experiment_types import (
        Action,
        Phase,
        SelectionDecision,
        SelectorSummary,
        StrategyOutcome,
    )
except ImportError:
    from experiment_types import (
        Action,
        Phase,
        SelectionDecision,
        SelectorSummary,
        StrategyOutcome,
    )


ACTION_ORDER = (Action.A1, Action.A2, Action.A3, Action.A4)


def _plain_decision(action):  # type: (Action) -> SelectionDecision
    if not isinstance(action, Action):
        raise TypeError("action must be an Action")
    return SelectionDecision(requested_action=action, applied_action=action)


class ManualSelector:
    def __init__(
        self,
        action=None,  # type: Optional[Action]
        sequence=None,  # type: Optional[Sequence[Action]]
        expected_windows=25,  # type: int
    ):
        if (action is None) == (sequence is None):
            raise ValueError("provide exactly one of action or sequence")
        if action is not None and not isinstance(action, Action):
            raise TypeError("action must be an Action")
        if sequence is not None:
            sequence = tuple(sequence)
            if len(sequence) != expected_windows:
                raise ValueError(
                    "manual sequence must contain exactly {} actions".format(
                        expected_windows
                    )
                )
            if not all(isinstance(item, Action) for item in sequence):
                raise TypeError("every manual sequence item must be an Action")
        self._action = action
        self._sequence = sequence

    def select(self, summary, rng):
        # type: (SelectorSummary, np.random.Generator) -> SelectionDecision
        if self._sequence is None:
            action = self._action
        else:
            if summary.window_index < 0 or summary.window_index >= len(self._sequence):
                raise IndexError("manual sequence window is out of range")
            action = self._sequence[summary.window_index]
        return _plain_decision(action)

    def observe(self, action, outcome):
        # type: (Action, StrategyOutcome) -> None
        return None


class UniformRandomSelector:
    def select(self, summary, rng):
        # type: (SelectorSummary, np.random.Generator) -> SelectionDecision
        index = int(rng.integers(0, len(ACTION_ORDER)))
        return _plain_decision(ACTION_ORDER[index])

    def observe(self, action, outcome):
        # type: (Action, StrategyOutcome) -> None
        return None


class RuleSelector:
    def select(self, summary, rng):
        # type: (SelectorSummary, np.random.Generator) -> SelectionDecision
        if summary.phase == Phase.FEASIBILITY:
            return _plain_decision(Action.A4)
        if summary.budget_fraction >= 0.8:
            return _plain_decision(Action.A3)
        if summary.stagnation_iterations >= 20:
            return _plain_decision(Action.A2)
        return _plain_decision(Action.A1)

    def observe(self, action, outcome):
        # type: (Action, StrategyOutcome) -> None
        return None


class UCB1Selector:
    def __init__(self):
        self._phase = None  # type: Optional[Phase]
        self._counts = {action: 0 for action in ACTION_ORDER}
        self._reward_sums = {action: 0.0 for action in ACTION_ORDER}
        self._archives = {}  # type: Dict[Phase, Dict[str, Dict[str, float]]]
        self._pending_action = None  # type: Optional[Action]

    @property
    def phase(self):  # type: () -> Optional[Phase]
        return self._phase

    @property
    def archives(self):  # type: () -> Mapping[Phase, Mapping[str, Mapping[str, float]]]
        return dict(self._archives)

    def statistics(self):  # type: () -> Dict[str, Dict[str, float]]
        return _statistics_snapshot(self._counts, self._reward_sums)

    def _enter_phase(self, phase):  # type: (Phase) -> None
        if self._phase == phase:
            return
        if self._phase == Phase.COST and phase == Phase.FEASIBILITY:
            raise ValueError("UCB1 cannot return from cost to feasibility phase")
        if self._phase is not None:
            self._archives[self._phase] = self.statistics()
        self._phase = phase
        self._counts = {action: 0 for action in ACTION_ORDER}
        self._reward_sums = {action: 0.0 for action in ACTION_ORDER}

    def select(self, summary, rng):
        # type: (SelectorSummary, np.random.Generator) -> SelectionDecision
        if self._pending_action is not None:
            raise RuntimeError("observe must be called before the next UCB1 selection")
        self._enter_phase(summary.phase)
        for action in ACTION_ORDER:
            if self._counts[action] == 0:
                self._pending_action = action
                return _plain_decision(action)

        total_pulls = sum(self._counts.values())
        scores = {}
        for action in ACTION_ORDER:
            mean_reward = self._reward_sums[action] / self._counts[action]
            exploration = math.sqrt(
                2.0 * math.log(total_pulls) / self._counts[action]
            )
            scores[action] = mean_reward + exploration
        best_score = max(scores.values())
        action = next(item for item in ACTION_ORDER if scores[item] == best_score)
        self._pending_action = action
        return _plain_decision(action)

    def observe(self, action, outcome):
        # type: (Action, StrategyOutcome) -> None
        if self._pending_action is None:
            raise RuntimeError("UCB1 observation has no pending selection")
        if action != self._pending_action or outcome.applied_action != action:
            raise ValueError("UCB1 observation action does not match selection")
        if outcome.phase_at_start != self._phase:
            raise ValueError("UCB1 outcome phase does not match selection phase")
        reward = float(np.clip(outcome.relative_improvement, -1.0, 1.0))
        self._counts[action] += 1
        self._reward_sums[action] += reward
        self._pending_action = None


class PerformanceTracker:
    def __init__(self):
        self._phases = {
            phase: self._empty_phase() for phase in (Phase.FEASIBILITY, Phase.COST)
        }

    @staticmethod
    def _empty_phase():  # type: () -> Dict[str, Dict[Action, float]]
        return {
            "counts": {action: 0 for action in ACTION_ORDER},
            "reward_sums": {action: 0.0 for action in ACTION_ORDER},
            "feasibility_transitions": {action: 0 for action in ACTION_ORDER},
            "invalid_outputs": {action: 0 for action in ACTION_ORDER},
            "fallbacks": {action: 0 for action in ACTION_ORDER},
        }

    def observe(self, action, outcome):
        # type: (Action, StrategyOutcome) -> None
        phase_data = self._phases[outcome.phase_at_start]
        phase_data["counts"][action] += 1
        phase_data["reward_sums"][action] += float(
            np.clip(outcome.relative_improvement, -1.0, 1.0)
        )
        if outcome.first_feasible_appeared:
            phase_data["feasibility_transitions"][action] += 1
        if outcome.error_kind is not None:
            phase_data["invalid_outputs"][action] += 1
        if outcome.fallback_used:
            phase_data["fallbacks"][action] += 1

    def payload(self, current_phase, actions=ACTION_ORDER):
        # type: (Phase, Sequence[Action]) -> Dict[str, Any]
        current = self._phase_payload(current_phase, actions)
        archived = {}
        for phase in (Phase.FEASIBILITY, Phase.COST):
            if phase != current_phase and any(
                self._phases[phase]["counts"][action] for action in ACTION_ORDER
            ):
                archived[phase.value] = self._phase_payload(phase, actions)
        return {"current_phase": current, "archived_phases": archived}

    def _phase_payload(self, phase, actions):
        # type: (Phase, Sequence[Action]) -> Dict[str, Any]
        data = self._phases[phase]
        result = {"phase": phase.value, "actions": {}}
        for action in actions:
            count = int(data["counts"][action])
            result["actions"][action.value] = {
                "count": count,
                "mean_reward": (
                    float(data["reward_sums"][action]) / count if count else None
                ),
                "feasibility_transition_count": int(
                    data["feasibility_transitions"][action]
                ),
                "invalid_output_count": int(data["invalid_outputs"][action]),
                "fallback_count": int(data["fallbacks"][action]),
            }
        return result


class ReplaySelector:
    def __init__(self, records, expected_windows=25):
        # type: (Sequence[Mapping[str, Any]], Optional[int]) -> None
        self._records = tuple(dict(record) for record in records)
        if expected_windows is not None and len(self._records) != expected_windows:
            raise ValueError(
                "replay must contain exactly {} records".format(expected_windows)
            )
        for index, record in enumerate(self._records):
            if int(record.get("window_index", index)) != index:
                raise ValueError("replay window indexes must be contiguous from zero")
            _record_action(record, "applied_action", required=True)
        self._cursor = 0

    @classmethod
    def from_jsonl(cls, path, expected_windows=25):
        # type: (Path, Optional[int]) -> ReplaySelector
        records = []  # type: List[Mapping[str, Any]]
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "invalid replay JSON on line {}".format(line_number)
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError("each replay line must be a JSON object")
                records.append(record)
        return cls(records, expected_windows=expected_windows)

    def select(self, summary, rng):
        # type: (SelectorSummary, np.random.Generator) -> SelectionDecision
        if self._cursor >= len(self._records):
            raise IndexError("replay action sequence is exhausted")
        record = self._records[self._cursor]
        _validate_replay_summary(record, summary)
        requested = _record_action(record, "requested_action", required=False)
        applied = _record_action(record, "applied_action", required=True)
        decision = SelectionDecision(
            requested_action=requested,
            applied_action=applied,
            fallback_used=bool(record.get("fallback_used", False)),
            error_kind=None,
            elapsed_seconds=0.0,
            llm_call_id=record.get("llm_call_id"),
        )
        self._cursor += 1
        return decision

    def observe(self, action, outcome):
        # type: (Action, StrategyOutcome) -> None
        if outcome.window_index != self._cursor - 1:
            raise ValueError("replay observation does not match the pending window")
        record = self._records[outcome.window_index]
        expected = _record_action(record, "applied_action", required=True)
        if action != expected or outcome.applied_action != expected:
            raise ValueError("replay outcome action does not match record")
        _validate_replay_outcome(record, outcome)


def _statistics_snapshot(counts, reward_sums):
    # type: (Mapping[Action, int], Mapping[Action, float]) -> Dict[str, Dict[str, float]]
    return {
        "counts": {action.value: int(counts[action]) for action in ACTION_ORDER},
        "mean_rewards": {
            action.value: (
                float(reward_sums[action]) / counts[action]
                if counts[action]
                else 0.0
            )
            for action in ACTION_ORDER
        },
    }


def _record_action(record, key, required):
    # type: (Mapping[str, Any], str, bool) -> Optional[Action]
    value = record.get(key)
    if value is None:
        if required:
            raise ValueError("replay record is missing {}".format(key))
        return None
    try:
        return Action(value)
    except ValueError as error:
        raise ValueError("invalid replay action {!r}".format(value)) from error


def _validate_replay_summary(record, summary):
    # type: (Mapping[str, Any], SelectorSummary) -> None
    expected = {
        "window_index": summary.window_index,
        "start_iteration": summary.iteration,
        "start_nfe": summary.search_nfe,
        "phase_at_start": summary.phase.value,
        "start_best_cost": summary.global_best_cost,
        "start_best_cv": summary.global_best_cv,
    }
    for key, actual in expected.items():
        if key in record and record[key] != actual:
            raise ValueError(
                "replay {} mismatch: expected {!r}, got {!r}".format(
                    key, record[key], actual
                )
            )


def _validate_replay_outcome(record, outcome):
    # type: (Mapping[str, Any], StrategyOutcome) -> None
    expected = {
        "window_index": outcome.window_index,
        "phase_at_start": outcome.phase_at_start.value,
        "phase_at_end": outcome.phase_at_end.value,
        "start_iteration": outcome.start_iteration,
        "end_iteration": outcome.end_iteration,
        "start_nfe": outcome.start_nfe,
        "end_nfe": outcome.end_nfe,
        "start_best_cost": outcome.start_best_cost,
        "end_best_cost": outcome.end_best_cost,
        "start_best_cv": outcome.start_best_cv,
        "end_best_cv": outcome.end_best_cv,
        "first_feasible_appeared": outcome.first_feasible_appeared,
        "first_feasible_nfe": outcome.first_feasible_nfe,
        "relative_improvement": outcome.relative_improvement,
        "requested_action": (
            outcome.requested_action.value
            if outcome.requested_action is not None
            else None
        ),
        "applied_action": outcome.applied_action.value,
        "fallback_used": outcome.fallback_used,
        "error_kind": (
            outcome.error_kind.value if outcome.error_kind is not None else None
        ),
        "llm_call_id": outcome.llm_call_id,
    }
    for key, actual in expected.items():
        if key in record and record[key] != actual:
            raise ValueError(
                "replay outcome {} mismatch: expected {!r}, got {!r}".format(
                    key, record[key], actual
                )
            )


def create_selector(spec, llm_provider=None):
    """Build the selector described by a RunSpec-like object."""
    fixed_actions = {
        "A1-only": Action.A1,
        "A2-only": Action.A2,
        "A3-only": Action.A3,
        "A4-only": Action.A4,
    }
    method = spec.method
    if method in fixed_actions:
        return ManualSelector(action=fixed_actions[method])
    if method == "UniformRandom":
        return UniformRandomSelector()
    if method == "Rule":
        return RuleSelector()
    if method == "UCB1":
        return UCB1Selector()
    if method not in ("LLM-E", "LLM-EP"):
        raise ValueError("Unknown method: {}".format(method))

    try:
        from .llm_selectors import (
            LLMConfig,
            LLMConfigurationError,
            LLMEPSelector,
            LLMESelector,
        )
    except ImportError:
        from llm_selectors import (
            LLMConfig,
            LLMConfigurationError,
            LLMEPSelector,
            LLMESelector,
        )

    if llm_provider is None:
        raise LLMConfigurationError(
            "{} requires an injected LLM provider".format(method)
        )
    selector_options = dict(getattr(spec, "selector_options", {}) or {})
    unknown_options = sorted(set(selector_options) - {"unavailable_actions"})
    if unknown_options:
        raise LLMConfigurationError(
            "unsupported selector options: {}".format(", ".join(unknown_options))
        )
    config = LLMConfig(**dict(getattr(spec, "llm_options", {}) or {}))
    selector_class = LLMESelector if method == "LLM-E" else LLMEPSelector
    return selector_class(
        provider=llm_provider,
        config=config,
        unavailable_actions=selector_options.get("unavailable_actions", ()),
    )
