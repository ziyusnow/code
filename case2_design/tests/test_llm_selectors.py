from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


CASE2_DIR = Path(__file__).resolve().parents[1]
if str(CASE2_DIR) not in sys.path:
    sys.path.insert(0, str(CASE2_DIR))

from experiment_types import Action, Phase, SelectorErrorKind
from llm_selectors import (
    LLMConfig,
    LLMConfigurationError,
    LLMEPSelector,
    LLMESelector,
    ProviderResponse,
    parse_action_response,
)
from strategy_selectors import create_selector
from test_selectors import make_outcome, make_summary


class ScriptedProvider:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def complete(self, request, config):
        self.calls.append((request, config))
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def config(**overrides):
    values = {
        "provider": "mock",
        "model_id": "model",
        "model_version_or_snapshot": "snapshot",
        "prompt_version": "prompt-v1",
        "response_schema_version": "schema-v1",
        "timeout_seconds": 5.0,
        "temperature": 0.0,
    }
    values.update(overrides)
    return LLMConfig(**values)


class LLMSelectorTests(unittest.TestCase):
    def test_configuration_is_validated_before_any_run(self):
        with self.assertRaises(LLMConfigurationError):
            config(model_id="")
        with self.assertRaises(LLMConfigurationError):
            config(temperature=0.1)
        with self.assertRaises(LLMConfigurationError):
            config(timeout_seconds=0)
        with self.assertRaises(LLMConfigurationError):
            LLMESelector(None, config())

    def test_valid_response_calls_provider_once_and_logs_record(self):
        provider = ScriptedProvider(
            [
                ProviderResponse(
                    '{"action":"A3"}',
                    provider_request_id="request-1",
                    usage={"total_tokens": 12},
                    cost=0.02,
                )
            ]
        )
        selector = LLMESelector(provider, config())
        decision = selector.select(make_summary(), np.random.default_rng(1))
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(decision.requested_action, Action.A3)
        self.assertEqual(decision.applied_action, Action.A3)
        self.assertFalse(decision.fallback_used)
        record = selector.call_records[0]
        self.assertEqual(record["provider_request_id"], "request-1")
        self.assertEqual(record["usage"]["total_tokens"], 12)
        self.assertEqual(record["provider_cost"], 0.02)
        self.assertEqual(set(record["request"]["response_schema"]["required"]), {"action"})

    def test_strict_single_key_json_parser(self):
        self.assertEqual(parse_action_response('{"action":"A1"}'), Action.A1)
        invalid = (
            "not json",
            "[]",
            '{"action":"A1","reason":"extra"}',
            '{"action":1}',
            '{"action":"A9"}',
        )
        for raw in invalid:
            with self.assertRaises(ValueError):
                parse_action_response(raw)

    def test_error_classes_single_call_and_phase_fallbacks(self):
        cases = (
            (TimeoutError("late"), SelectorErrorKind.PROVIDER_TIMEOUT),
            (RuntimeError("offline"), SelectorErrorKind.PROVIDER_ERROR),
            (ProviderResponse("not json"), SelectorErrorKind.INVALID_JSON),
            (ProviderResponse("[]"), SelectorErrorKind.SCHEMA_ERROR),
            (ProviderResponse('{"action":"A9"}'), SelectorErrorKind.INVALID_ACTION),
        )
        for scripted, kind in cases:
            with self.subTest(kind=kind):
                provider = ScriptedProvider([scripted])
                selector = LLMESelector(provider, config())
                decision = selector.select(make_summary(), np.random.default_rng(1))
                self.assertEqual(len(provider.calls), 1)
                self.assertEqual(decision.error_kind, kind)
                self.assertTrue(decision.fallback_used)
                self.assertEqual(decision.applied_action, Action.A4)

        provider = ScriptedProvider([ProviderResponse("[]")])
        selector = LLMESelector(provider, config())
        decision = selector.select(
            make_summary(phase=Phase.COST), np.random.default_rng(1)
        )
        self.assertEqual(decision.applied_action, Action.A1)

    def test_unavailable_action_changes_request_validation_and_fallback(self):
        provider = ScriptedProvider([ProviderResponse('{"action":"A4"}')])
        selector = LLMEPSelector(
            provider,
            config(),
            unavailable_actions=(Action.A4,),
        )
        decision = selector.select(make_summary(), np.random.default_rng(1))
        request = provider.calls[0][0]
        self.assertEqual(request.response_schema["properties"]["action"]["enum"], [
            "A1", "A2", "A3"
        ])
        self.assertEqual(request.payload["allowed_actions"], ["A1", "A2", "A3"])
        self.assertNotIn("A4", request.payload["action_definitions"])
        self.assertNotIn(
            "A4", request.payload["action_performance"]["current_phase"]["actions"]
        )
        self.assertNotIn("A4", request.system_instruction)
        self.assertEqual(decision.error_kind, SelectorErrorKind.INVALID_ACTION)
        self.assertTrue(decision.fallback_used)
        self.assertEqual(decision.applied_action, Action.A1)

    def test_unavailable_actions_are_validated(self):
        provider = ScriptedProvider([])
        with self.assertRaises(LLMConfigurationError):
            LLMESelector(provider, config(), unavailable_actions=("A9",))
        with self.assertRaises(LLMConfigurationError):
            LLMESelector(
                provider,
                config(),
                unavailable_actions=tuple(Action),
            )

    def test_bad_provider_return_is_provider_error(self):
        provider = ScriptedProvider([{"action": "A1"}])
        selector = LLMESelector(provider, config())
        decision = selector.select(make_summary(), np.random.default_rng(1))
        self.assertEqual(decision.error_kind, SelectorErrorKind.PROVIDER_ERROR)
        self.assertEqual(len(provider.calls), 1)

    def test_selector_implementation_error_propagates(self):
        provider = ScriptedProvider([])
        selector = LLMESelector(provider, config())

        def fail_to_build_payload(summary):
            raise RuntimeError("selector implementation failed")

        selector._build_payload = fail_to_build_payload
        with self.assertRaisesRegex(RuntimeError, "selector implementation failed"):
            selector.select(make_summary(), np.random.default_rng(1))
        self.assertEqual(provider.calls, [])

    def test_llm_e_excludes_performance_and_llm_ep_separates_phases(self):
        e_provider = ScriptedProvider([ProviderResponse('{"action":"A1"}')])
        e_selector = LLMESelector(e_provider, config())
        e_selector.select(make_summary(), np.random.default_rng(1))
        e_payload = e_provider.calls[0][0].payload
        self.assertNotIn("action_performance", e_payload)

        ep_provider = ScriptedProvider(
            [ProviderResponse('{"action":"A4"}'), ProviderResponse('{"action":"A1"}')]
        )
        ep_selector = LLMEPSelector(ep_provider, config())
        feasibility_summary = make_summary()
        decision = ep_selector.select(feasibility_summary, np.random.default_rng(1))
        ep_selector.observe(
            decision.applied_action,
            make_outcome(
                decision.applied_action,
                phase=Phase.FEASIBILITY,
                phase_at_end=Phase.COST,
                first_feasible_appeared=True,
                first_feasible_nfe=12600,
            ),
        )
        ep_selector.select(
            make_summary(1, phase=Phase.COST), np.random.default_rng(1)
        )
        performance = ep_provider.calls[1][0].payload["action_performance"]
        self.assertEqual(performance["current_phase"]["phase"], "cost")
        archived = performance["archived_phases"]["feasibility"]
        self.assertEqual(archived["actions"]["A4"]["count"], 1)
        self.assertEqual(
            archived["actions"]["A4"]["feasibility_transition_count"], 1
        )
        self.assertIsNone(
            performance["current_phase"]["actions"]["A1"]["mean_reward"]
        )

    def test_selector_and_provider_do_not_consume_numpy_rng(self):
        provider = ScriptedProvider([ProviderResponse('{"action":"A2"}')])
        selector = LLMESelector(provider, config())
        selector_rng = np.random.default_rng(30)
        control = np.random.default_rng(30)
        selector.select(make_summary(), selector_rng)
        np.testing.assert_array_equal(selector_rng.random(20), control.random(20))

    def test_create_selector_uses_injected_provider_and_llm_config(self):
        provider = ScriptedProvider([ProviderResponse('{"action":"A2"}')])
        spec = SimpleNamespace(
            method="LLM-EP",
            selector_options={"unavailable_actions": ["A3"]},
            llm_options={
                "provider": "mock",
                "model_id": "model",
                "model_version_or_snapshot": "snapshot",
                "prompt_version": "prompt-v1",
                "response_schema_version": "schema-v1",
                "timeout_seconds": 5.0,
                "temperature": 0.0,
            },
        )
        with self.assertRaises(LLMConfigurationError):
            create_selector(spec)
        selector = create_selector(spec, llm_provider=provider)
        self.assertIsInstance(selector, LLMEPSelector)
        self.assertEqual(selector.allowed_actions, (Action.A1, Action.A2, Action.A4))
        decision = selector.select(make_summary(), np.random.default_rng(1))
        self.assertEqual(decision.applied_action, Action.A2)


if __name__ == "__main__":
    unittest.main()
