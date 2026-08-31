import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple

import numpy as np

try:
    from .experiment_types import (
        Action,
        Phase,
        SelectionDecision,
        SelectorErrorKind,
        SelectorSummary,
        StrategyOutcome,
    )
    from .strategy_selectors import ACTION_ORDER, PerformanceTracker
except ImportError:
    from experiment_types import (
        Action,
        Phase,
        SelectionDecision,
        SelectorErrorKind,
        SelectorSummary,
        StrategyOutcome,
    )
    from strategy_selectors import ACTION_ORDER, PerformanceTracker


class LLMConfigurationError(ValueError):
    pass


class InvalidJSONError(ValueError):
    pass


class ResponseSchemaError(ValueError):
    pass


class InvalidActionError(ValueError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model_id: str
    model_version_or_snapshot: str
    prompt_version: str
    response_schema_version: str
    timeout_seconds: float
    temperature: float = 0.0

    def __post_init__(self):
        required = (
            "provider",
            "model_id",
            "model_version_or_snapshot",
            "prompt_version",
            "response_schema_version",
        )
        for field_name in required:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise LLMConfigurationError("{} must be specified".format(field_name))
        if self.temperature != 0:
            raise LLMConfigurationError("formal LLM experiments require temperature=0")
        if self.timeout_seconds <= 0:
            raise LLMConfigurationError("timeout_seconds must be positive")


@dataclass(frozen=True)
class ProviderRequest:
    call_id: str
    system_instruction: str
    payload: Mapping[str, Any]
    response_schema: Mapping[str, Any]


@dataclass(frozen=True)
class ProviderResponse:
    raw_text: str
    provider_request_id: Optional[str] = None
    usage: Optional[Mapping[str, Any]] = None
    cost: Optional[float] = None


class LLMProvider(Protocol):
    def complete(
        self, request: ProviderRequest, config: LLMConfig
    ) -> ProviderResponse:
        ...


ACTION_DEFINITIONS = {
    "A1": "Use the unmodified MPPSO candidate population.",
    "A2": "Apply DE-inspired exploration to the worst-ranked particles.",
    "A3": "Apply shrinking perturbations around the current global best.",
    "A4": "Move high-violation particles toward the best reference pbest.",
}


def _normalize_unavailable_actions(values):
    unavailable = set()
    for value in values:
        try:
            unavailable.add(value if isinstance(value, Action) else Action(value))
        except (TypeError, ValueError) as error:
            raise LLMConfigurationError(
                "unavailable_actions contains an invalid action {!r}".format(value)
            ) from error
    return unavailable


def _response_schema(allowed_actions):
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [action.value for action in allowed_actions],
            }
        },
        "required": ["action"],
        "additionalProperties": False,
    }


def _system_instruction(allowed_actions):
    choices = ", ".join(action.value for action in allowed_actions)
    return (
        "Select exactly one Case 2 search action from: {}. Return only a JSON "
        "object with the single key action."
    ).format(choices)


def _fallback_action(phase, allowed_actions):
    if phase == Phase.FEASIBILITY and Action.A4 in allowed_actions:
        return Action.A4
    if Action.A1 in allowed_actions:
        return Action.A1
    return allowed_actions[0]

def parse_action_response(raw_text, allowed_actions=ACTION_ORDER):
    # type: (str, Tuple[Action, ...]) -> Action
    if not isinstance(raw_text, str):
        raise ResponseSchemaError("provider raw_text must be a string")
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise InvalidJSONError("response is not valid JSON") from error
    if not isinstance(value, dict):
        raise ResponseSchemaError("response must be a JSON object")
    if set(value.keys()) != {"action"} or not isinstance(value["action"], str):
        raise ResponseSchemaError("response must contain only a string action field")
    try:
        action = Action(value["action"])
    except ValueError as error:
        raise InvalidActionError("response action is not A1, A2, A3, or A4") from error
    if action not in allowed_actions:
        raise InvalidActionError("response action is unavailable for this run")
    return action


class _BaseLLMSelector:
    method_name = "LLM"

    def __init__(self, provider, config, unavailable_actions=()):
        # type: (LLMProvider, LLMConfig, Tuple[Action, ...]) -> None
        if not isinstance(config, LLMConfig):
            raise LLMConfigurationError("config must be an LLMConfig")
        if provider is None or not callable(getattr(provider, "complete", None)):
            raise LLMConfigurationError("provider must implement complete(request, config)")
        self.provider = provider
        self.config = config
        unavailable = _normalize_unavailable_actions(unavailable_actions)
        self.allowed_actions = tuple(
            action for action in ACTION_ORDER if action not in unavailable
        )
        if not self.allowed_actions:
            raise LLMConfigurationError("at least one action must remain available")
        self._call_records = []  # type: list

    @property
    def call_records(self):  # type: () -> Tuple[Mapping[str, Any], ...]
        return tuple(dict(record) for record in self._call_records)

    def _extra_payload(self, summary):  # type: (SelectorSummary) -> Mapping[str, Any]
        return {}

    def _build_payload(self, summary):  # type: (SelectorSummary) -> Dict[str, Any]
        payload = {
            "method": self.method_name,
            "allowed_actions": [action.value for action in self.allowed_actions],
            "action_definitions": {
                action.value: ACTION_DEFINITIONS[action.value]
                for action in self.allowed_actions
            },
            "selector_summary": _summary_payload(summary),
        }
        payload.update(self._extra_payload(summary))
        return payload

    def select(self, summary, rng):
        # type: (SelectorSummary, np.random.Generator) -> SelectionDecision
        started = time.perf_counter()
        call_id = "{}-window-{:02d}".format(
            self.method_name.lower().replace("-", "_"), summary.window_index
        )
        raw_text = None
        response = None
        request = None
        requested_action = None
        error_kind = None
        error_message = None
        request = ProviderRequest(
            call_id=call_id,
            system_instruction=_system_instruction(self.allowed_actions),
            payload=self._build_payload(summary),
            response_schema=_response_schema(self.allowed_actions),
        )
        try:
            provider_response = self.provider.complete(request, self.config)
            if not isinstance(provider_response, ProviderResponse):
                raise TypeError("provider must return ProviderResponse")
            if (
                provider_response.usage is not None
                and not isinstance(provider_response.usage, Mapping)
            ):
                raise TypeError("ProviderResponse.usage must be a mapping or None")
            response = provider_response
            raw_text = response.raw_text
        except TimeoutError as error:
            error_kind = SelectorErrorKind.PROVIDER_TIMEOUT
            error_message = str(error)
        except Exception as error:
            error_kind = SelectorErrorKind.PROVIDER_ERROR
            error_message = str(error)
        else:
            try:
                requested_action = parse_action_response(
                    raw_text, self.allowed_actions
                )
            except InvalidJSONError as error:
                error_kind = SelectorErrorKind.INVALID_JSON
                error_message = str(error)
            except ResponseSchemaError as error:
                error_kind = SelectorErrorKind.SCHEMA_ERROR
                error_message = str(error)
            except InvalidActionError as error:
                error_kind = SelectorErrorKind.INVALID_ACTION
                error_message = str(error)

        elapsed = time.perf_counter() - started
        fallback_used = error_kind is not None
        applied_action = requested_action
        if fallback_used:
            applied_action = _fallback_action(summary.phase, self.allowed_actions)
        decision = SelectionDecision(
            requested_action=requested_action,
            applied_action=applied_action,
            fallback_used=fallback_used,
            error_kind=error_kind,
            elapsed_seconds=elapsed,
            llm_call_id=call_id,
        )
        self._call_records.append(
            _call_record(
                self.method_name,
                self.config,
                summary,
                request,
                response,
                raw_text,
                decision,
                error_message,
            )
        )
        return decision

    def observe(self, action, outcome):
        # type: (Action, StrategyOutcome) -> None
        return None


class LLMESelector(_BaseLLMSelector):
    method_name = "LLM-E"


class LLMEPSelector(_BaseLLMSelector):
    method_name = "LLM-EP"

    def __init__(self, provider, config, unavailable_actions=()):
        super().__init__(provider, config, unavailable_actions=unavailable_actions)
        self.performance = PerformanceTracker()

    def _extra_payload(self, summary):
        # type: (SelectorSummary) -> Mapping[str, Any]
        return {
            "action_performance": self.performance.payload(
                summary.phase, self.allowed_actions
            )
        }

    def observe(self, action, outcome):
        # type: (Action, StrategyOutcome) -> None
        self.performance.observe(action, outcome)


def _summary_payload(summary):  # type: (SelectorSummary) -> Dict[str, Any]
    return {
        "window_index": summary.window_index,
        "iteration": summary.iteration,
        "search_nfe": summary.search_nfe,
        "budget_fraction": summary.budget_fraction,
        "phase": summary.phase.value,
        "global_best_cost": summary.global_best_cost,
        "global_best_cv": summary.global_best_cv,
        "cv_r": summary.cv_r,
        "cv_soc": summary.cv_soc,
        "cv_e": summary.cv_e,
        "cv_d": summary.cv_d,
        "feasible_fraction": summary.feasible_fraction,
        "population_cv_q25": summary.population_cv_q25,
        "population_cv_median": summary.population_cv_median,
        "population_cv_q75": summary.population_cv_q75,
        "normalized_diversity": summary.normalized_diversity,
        "stagnation_iterations": summary.stagnation_iterations,
        "previous_action": (
            summary.previous_action.value if summary.previous_action is not None else None
        ),
    }


def _call_record(
    method_name,
    config,
    summary,
    request,
    response,
    raw_text,
    decision,
    error_message,
):
    # type: (str, LLMConfig, SelectorSummary, Optional[ProviderRequest], Optional[ProviderResponse], Optional[str], SelectionDecision, Optional[str]) -> Dict[str, Any]
    return {
        "call_id": decision.llm_call_id,
        "method": method_name,
        "window_index": summary.window_index,
        "phase": summary.phase.value,
        "provider": config.provider,
        "model_id": config.model_id,
        "model_version_or_snapshot": config.model_version_or_snapshot,
        "prompt_version": config.prompt_version,
        "response_schema_version": config.response_schema_version,
        "temperature": config.temperature,
        "timeout_seconds": config.timeout_seconds,
        "request": (
            {
                "system_instruction": request.system_instruction,
                "payload": request.payload,
                "response_schema": request.response_schema,
            }
            if request is not None
            else None
        ),
        "raw_response": raw_text,
        "provider_request_id": (
            response.provider_request_id if response is not None else None
        ),
        "usage": dict(response.usage) if response is not None and response.usage else None,
        "provider_cost": response.cost if response is not None else None,
        "requested_action": (
            decision.requested_action.value
            if decision.requested_action is not None
            else None
        ),
        "applied_action": decision.applied_action.value,
        "fallback_used": decision.fallback_used,
        "error_kind": (
            decision.error_kind.value if decision.error_kind is not None else None
        ),
        "error_message": error_message,
        "elapsed_seconds": decision.elapsed_seconds,
    }
