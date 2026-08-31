import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


CASE2_DIR = Path(__file__).resolve().parents[1]
if str(CASE2_DIR) not in sys.path:
    sys.path.insert(0, str(CASE2_DIR))

from analyze_results import (
    analyze_experiment,
    collect_runs,
    exact_mcnemar,
    holm_adjust,
    llm_summary,
    paired_comparisons,
    paired_cost_statistics,
    step_auc,
    summarize_methods,
    render_figures,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def task(method, seed, status="pending"):
    method_id = method
    config = {
        "experiment_id": "synthetic",
        "group": "formal",
        "method": method,
        "method_id": method_id,
        "variant": "default",
        "seed": seed,
    }
    return {
        "task_id": "%s/seed_%d" % (method_id, seed),
        "method": method,
        "method_id": method_id,
        "variant": "default",
        "seed": seed,
        "relative_path": "%s/seed_%d" % (method_id, seed),
        "status": status,
        "config": config,
    }


def complete_run(root, method, seed, costs, first_feasible_nfe=10):
    directory = root / "synthetic" / method / ("seed_%d" % seed)
    config = task(method, seed, "complete")["config"]
    write_json(directory / "config.json", config)
    write_json(
        directory / "summary.json",
        {
            "status": "complete",
            "method_id": method,
            "seed": seed,
            "feasible": True,
            "best_cost": costs[-1],
            "first_feasible_nfe": first_feasible_nfe,
            "search_nfe": 30,
            "wall_seconds": 2.0,
            "selector_seconds": 0.2,
        },
    )
    pd.DataFrame(
        {
            "search_nfe": [10, 20, 30],
            "best_cost": costs,
            "feasible": [True, True, True],
        }
    ).to_csv(directory / "history.csv", index=False)
    write_jsonl(
        directory / "actions.jsonl",
        [
            {
                "applied_action": "A1",
                "phase_at_start": "cost",
                "relative_improvement": 0.1,
            }
        ],
    )


class AnalysisStatisticsTests(unittest.TestCase):
    def test_exact_mcnemar_and_holm(self):
        result = exact_mcnemar(
            [True, True, True, False], [False, False, False, False]
        )
        self.assertEqual(result["discordant"], 3)
        self.assertAlmostEqual(result["p_value"], 0.25)
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": None}, family_size=3)
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.08)
        self.assertIsNone(adjusted["c"])

    def test_fewer_than_ten_pairs_are_descriptive_only(self):
        result = paired_cost_statistics(np.arange(1.0, 10.0), np.arange(2.0, 11.0))
        self.assertFalse(result["cost_inference_available"])
        self.assertEqual(result["cost_analysis_status"], "descriptive_only")
        self.assertTrue(np.isnan(result["difference_ci_low"]))
        self.assertTrue(np.isnan(result["relative_ci_high"]))
        self.assertIsNone(result["wilcoxon_p"])

    def test_ten_pairs_include_bootstrap_and_wilcoxon(self):
        result = paired_cost_statistics(np.arange(1.0, 11.0), np.arange(2.0, 12.0))
        self.assertTrue(result["cost_inference_available"])
        self.assertTrue(np.isfinite(result["difference_ci_low"]))
        self.assertAlmostEqual(result["median_difference"], -1.0)
        self.assertIsNotNone(result["wilcoxon_p"])

    def test_auc_excludes_history_before_requested_start(self):
        history = pd.DataFrame(
            {"search_nfe": [10, 20, 30], "best_cost": [100.0, 10.0, 8.0]}
        )
        self.assertAlmostEqual(step_auc(history, 20, 30), 10.0)

    def test_feasible_rate_figure_handles_all_successes(self):
        output = CASE2_DIR / "analysis_figure_test"
        self.addCleanup(lambda: shutil.rmtree(output, ignore_errors=True))
        methods = pd.DataFrame(
            [
                {
                    "method_id": "A1-only",
                    "feasible_rate": 1.0,
                    "wilson_low": 0.886486606826031,
                    "wilson_high": 0.9999999999999999,
                }
            ]
        )
        render_figures(
            pd.DataFrame(columns=("resolved", "feasible", "path")),
            methods,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            output,
        )
        self.assertTrue((output / "feasible_rate.png").exists())


class AnalysisIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="analysis-test-", dir=CASE2_DIR))
        self.results_root = self.temp_dir / "results"
        experiment_root = self.results_root / "synthetic"
        tasks = [
            task("A1-only", 1, "complete"),
            task("A2-only", 1, "complete"),
            task("A1-only", 2),
            task("A2-only", 2),
        ]
        write_json(
            experiment_root / "manifest.json",
            {
                "schema_version": 1,
                "experiment_id": "synthetic",
                "group": "formal",
                "expected_task_count": len(tasks),
                "tasks": tasks,
            },
        )
        complete_run(self.results_root, "A1-only", 1, [12.0, 9.0, 9.0])
        complete_run(self.results_root, "A2-only", 1, [10.0, 8.0, 8.0])

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_manifest_preserves_unstarted_denominators(self):
        runs = collect_runs("synthetic", self.results_root)
        self.assertEqual(len(runs), 4)
        self.assertEqual(int(runs["resolved"].sum()), 2)
        self.assertEqual(int(runs["unstarted"].sum()), 2)
        summary = summarize_methods(runs).set_index("method_id")
        self.assertEqual(summary.loc["A1-only", "planned"], 2)
        self.assertEqual(summary.loc["A1-only", "resolved"], 1)
        self.assertEqual(summary.loc["A1-only", "unstarted"], 1)

    def test_analysis_rejects_output_outside_case2_directory(self):
        outside = CASE2_DIR.parent / "analysis-outside-must-not-exist"
        self.assertFalse(outside.exists())
        with self.assertRaisesRegex(ValueError, "must stay under"):
            analyze_experiment("synthetic", outside)
        self.assertFalse(outside.exists())

    def test_pairwise_auc_uses_shared_feasible_interval(self):
        runs = collect_runs("synthetic", self.results_root)
        paired = paired_comparisons(runs, comparisons=(("A2-only", "A1-only"),))
        row = paired.iloc[0]
        self.assertEqual(row["auc_joint_feasible_pairs"], 1)
        self.assertAlmostEqual(row["auc_candidate_median"], 9.0)
        self.assertAlmostEqual(row["auc_reference_median"], 10.5)
        self.assertAlmostEqual(row["auc_median_difference"], -1.5)
        self.assertEqual(row["auc_cost_analysis_status"], "descriptive_only")
        self.assertTrue(np.isnan(row["auc_difference_ci_low"]))

    def test_end_to_end_writes_secondary_tables_and_figures(self):
        analyze_experiment("synthetic", self.results_root)
        aggregate = self.results_root / "synthetic" / "aggregate"
        for name in (
            "runs.csv",
            "method_summary.csv",
            "pairwise_stats.csv",
            "convergence_nfe.csv",
            "phase_rewards.csv",
            "selector_overhead.csv",
            "llm_metrics.csv",
            "report.md",
        ):
            self.assertTrue((aggregate / name).exists(), name)
        for name in (
            "feasible_rate.png",
            "final_cost.png",
            "convergence_nfe.png",
            "phase_rewards.png",
            "selector_overhead.png",
        ):
            self.assertTrue((aggregate / "figures" / name).exists(), name)
        report = (aggregate / "report.md").read_text(encoding="utf-8")
        self.assertIn("Resolved runs: `2/4`", report)
        self.assertIn("Unstarted runs: `2`", report)

    def test_all_pending_manifest_can_be_analyzed(self):
        pending = task("A1-only", 9)
        pending["config"]["experiment_id"] = "pending"
        write_json(
            self.results_root / "pending" / "manifest.json",
            {
                "schema_version": 1,
                "experiment_id": "pending",
                "group": "baseline",
                "expected_task_count": 1,
                "tasks": [pending],
            },
        )
        runs, summary, pairwise = analyze_experiment(
            "pending", self.results_root, diagnostic=True
        )
        self.assertEqual(len(runs), 1)
        self.assertEqual(int(runs["resolved"].sum()), 0)
        self.assertEqual(summary.iloc[0]["planned"], 1)
        self.assertTrue(pairwise.empty)

    def test_llm_totals_include_failures_tokens_cost_and_time(self):
        llm_dir = self.results_root / "synthetic" / "LLM-E" / "seed_3"
        write_jsonl(
            llm_dir / "llm_calls.jsonl",
            [
                {
                    "fallback_used": True,
                    "error_kind": "invalid_schema",
                    "usage": {"prompt_tokens": 11, "completion_tokens": 4},
                    "provider_cost": 0.02,
                    "elapsed_seconds": 0.5,
                },
                {
                    "fallback_used": True,
                    "error_kind": "timeout",
                    "usage": {"total_tokens": 3},
                    "elapsed_seconds": 1.0,
                },
            ],
        )
        runs = pd.DataFrame(
            [
                {
                    "resolved": True,
                    "method_id": "LLM-E",
                    "path": str(llm_dir),
                }
            ]
        )
        row = llm_summary(runs).iloc[0]
        self.assertEqual(row["calls"], 2)
        self.assertEqual(row["fallbacks"], 2)
        self.assertEqual(row["invalid_outputs"], 1)
        self.assertEqual(row["timeouts"], 1)
        self.assertEqual(row["total_tokens"], 18)
        self.assertAlmostEqual(row["provider_cost"], 0.02)
        self.assertAlmostEqual(row["elapsed_seconds"], 1.5)


if __name__ == "__main__":
    unittest.main()
