import argparse
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CASE2_DIR = Path(__file__).resolve().parents[1]
if str(CASE2_DIR) not in sys.path:
    sys.path.insert(0, str(CASE2_DIR))

import experiment_io
import run_experiments
import solve_case2
from experiment_io import RunSpec
from experiment_types import SearchConfig


class RunnerIOTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="runner_test_", dir=str(CASE2_DIR)))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_atomic_write_via_commits_on_windows_compatible_descriptor(self):
        destination = self.temp_dir / "artifact.txt"

        def writer(path):
            Path(path).write_text("committed", encoding="utf-8")

        experiment_io.atomic_write_via(destination, writer)
        self.assertEqual(destination.read_text(encoding="utf-8"), "committed")

    def test_atomic_write_retries_transient_windows_replace_failure(self):
        destination = self.temp_dir / "manifest.json"
        real_replace = experiment_io.os.replace
        calls = []

        def transient_replace(source, target):
            calls.append((source, target))
            if len(calls) == 1:
                raise PermissionError("destination is briefly in use")
            return real_replace(source, target)

        with mock.patch.object(experiment_io.os, "replace", transient_replace), mock.patch.object(
            experiment_io.time, "sleep"
        ) as sleep:
            experiment_io.atomic_write_json(destination, {"status": "complete"})

        self.assertEqual(experiment_io.read_json(destination), {"status": "complete"})
        self.assertEqual(len(calls), 2)
        sleep.assert_called_once_with(0.05)

    def test_provider_name_is_serialized_but_runtime_provider_is_separate(self):
        runtime_provider = mock.Mock()
        spec = RunSpec(
            experiment_id="provider_redaction",
            group="validation",
            method="LLM-E",
            seed=1,
            llm_options={
                "provider": "mock",
                "model_id": "test-model",
                "model_version_or_snapshot": "v1",
                "prompt_version": "p1",
                "response_schema_version": "s1",
                "timeout_seconds": 1.0,
            },
        )
        payload = spec.to_dict()
        self.assertEqual(payload["llm_options"]["provider"], "mock")
        selector = run_experiments._create_selector(spec, runtime_provider)
        self.assertIs(selector.provider, runtime_provider)
        self.assertNotIn(str(runtime_provider), json.dumps(payload))
        json.dumps(payload)

    def test_default_task_matrices_and_non_llm_alias(self):
        expected = {
            "baseline": (1, 1, 1),
            "validation": (45, 9, 5),
            "formal": (270, 9, 30),
            "diagnostic": (170, 17, 10),
        }
        for group, (task_count, method_count, seed_count) in expected.items():
            with self.subTest(group=group):
                specs = run_experiments.build_run_specs(group, "matrix_%s" % group)
                self.assertEqual(len(specs), task_count)
                self.assertEqual(len({spec.method_id for spec in specs}), method_count)
                self.assertEqual(len({spec.seed for spec in specs}), seed_count)

        specs = run_experiments.build_run_specs(
            "formal", "non_llm", methods=("non-llm",)
        )
        self.assertEqual(len(specs), 7 * 30)
        self.assertEqual(
            {spec.method for spec in specs}, set(run_experiments.NON_LLM_METHODS)
        )

    def test_manifest_exposes_all_pending_tasks_before_run_directories_exist(self):
        specs = run_experiments.build_run_specs(
            "validation", "pending_manifest", methods=("A1-only",), seeds=(1, 2)
        )
        manifest_path = experiment_io.initialize_experiment_manifest(
            self.temp_dir, specs, "2026-08-21T00:00:00+00:00"
        )
        manifest = experiment_io.read_json(manifest_path)
        self.assertEqual(manifest["expected_task_count"], 2)
        self.assertEqual(manifest["expected_method_count"], 1)
        self.assertEqual(manifest["expected_seed_count"], 2)
        self.assertEqual([task["status"] for task in manifest["tasks"]], ["pending", "pending"])
        self.assertFalse((manifest_path.parent / "A1-only" / "seed_1" / "config.json").exists())

        rows = run_experiments.experiment_status("pending_manifest", self.temp_dir)
        self.assertEqual([row["state"] for row in rows], ["pending", "pending"])

    def test_run_command_rejects_llm_without_provider_before_manifest(self):
        args = argparse.Namespace(
            group="validation",
            experiment_id="missing_provider",
            methods="LLM-E",
            seeds="1",
            llm_config=None,
            results_root=str(self.temp_dir),
            resume=False,
            jobs=1,
        )
        with self.assertRaises(run_experiments.LLMConfigurationError):
            run_experiments._run_command(args)
        self.assertFalse((self.temp_dir / "missing_provider" / "manifest.json").exists())

    def test_output_path_outside_case2_is_rejected_without_writing(self):
        outside = CASE2_DIR.parent / "not_allowed_results"
        with self.assertRaises(ValueError):
            run_experiments._require_case2_path(outside)
        self.assertFalse(outside.exists())

    def test_validation_exception_is_recorded_as_numerical_failure(self):
        config = SearchConfig(population_size=8, max_iterations=1, decision_interval=1)
        spec = RunSpec(
            experiment_id="numerical_failure",
            group="validation",
            method="A1-only",
            seed=91,
            search_config=config,
        )
        with mock.patch.object(
            solve_case2, "validation_metrics", side_effect=FloatingPointError("nonfinite audit")
        ):
            outcome = run_experiments.run_one(spec, self.temp_dir)

        self.assertEqual(outcome["state"], "complete")
        self.assertEqual(outcome["summary"]["outcome"], "numerical_failure")
        run_dir = Path(outcome["path"])
        validation = experiment_io.read_json(run_dir / "validation.json")
        self.assertIn("FloatingPointError", validation["validation_error"])
        self.assertFalse((run_dir / "attempts.jsonl").exists())
        self.assertTrue((run_dir / "summary.json").exists())

    def test_small_real_run_commits_all_artifacts_and_exact_nfe(self):
        config = SearchConfig(population_size=8, max_iterations=1, decision_interval=1)
        spec = RunSpec(
            experiment_id="small_real_run",
            group="validation",
            method="A1-only",
            seed=92,
            search_config=config,
        )
        outcome = run_experiments.run_one(spec, self.temp_dir)
        self.assertEqual(outcome["state"], "complete")
        self.assertEqual(outcome["summary"]["search_nfe"], 16)
        run_dir = Path(outcome["path"])
        for name in (
            "config.json",
            "history.csv",
            "actions.jsonl",
            "schedule.csv",
            "validation.json",
            "summary.json",
        ):
            self.assertTrue((run_dir / name).exists(), name)

    def test_run_command_updates_manifest_after_completed_task(self):
        args = argparse.Namespace(
            group="baseline",
            experiment_id="completed_manifest",
            methods="all",
            seeds="1",
            llm_config=None,
            results_root=str(self.temp_dir),
            resume=False,
            jobs=1,
        )
        fake = {
            "state": "complete",
            "path": "unused",
            "summary": {"outcome": "feasible"},
        }
        with mock.patch.object(run_experiments, "run_one", return_value=fake):
            self.assertEqual(run_experiments._run_command(args), 0)
        manifest = experiment_io.read_json(
            self.temp_dir / "completed_manifest" / "manifest.json"
        )
        self.assertEqual(manifest["tasks"][0]["status"], "complete")
        self.assertEqual(manifest["tasks"][0]["outcome"], "feasible")


if __name__ == "__main__":
    unittest.main()
