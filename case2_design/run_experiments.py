from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import argparse
import concurrent.futures
import sys
import time
import traceback

import numpy as np

try:
    from .experiment_io import (
        RunPaths,
        RunSpec,
        append_attempt,
        commit_run_artifacts,
        completed_summary,
        ensure_run_config,
        initialize_experiment_manifest,
        read_json,
        to_jsonable,
        update_manifest_task,
    )
    from .experiment_types import ActionParameters, SearchConfig
except ImportError:
    from experiment_io import (
        RunPaths,
        RunSpec,
        append_attempt,
        commit_run_artifacts,
        completed_summary,
        ensure_run_config,
        initialize_experiment_manifest,
        read_json,
        to_jsonable,
        update_manifest_task,
    )
    from experiment_types import ActionParameters, SearchConfig


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = BASE_DIR / "results"

FORMAL_METHODS = (
    "A1-only",
    "A2-only",
    "A3-only",
    "A4-only",
    "UniformRandom",
    "Rule",
    "UCB1",
    "LLM-E",
    "LLM-EP",
)
NON_LLM_METHODS = FORMAL_METHODS[:7]
BASELINE_SEEDS = (20260814,)
VALIDATION_SEEDS = tuple(range(300001, 300006))
FORMAL_SEEDS = tuple(range(310001, 310031))
DIAGNOSTIC_SEEDS = tuple(range(320001, 320011))

HISTORY_FIELDS = (
    "iteration",
    "search_nfe",
    "action",
    "phase",
    "best_cost",
    "best_cv",
    "best_fitness",
    "feasible",
    "feasible_fraction",
    "cv_r",
    "cv_soc",
    "cv_e",
    "cv_d",
    "median_cv",
    "normalized_diversity",
    "improved",
    "stagnation_iterations",
    "requested_action",
    "selector_fallback",
    "selector_error_kind",
    "selector_elapsed_seconds",
    "llm_call_id",
)


class LLMConfigurationError(RuntimeError):
    pass


def _utc_now():  # type: () -> str
    return datetime.now(timezone.utc).isoformat()


def _require_case2_path(path):  # type: (Path) -> Path
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(BASE_DIR)
    except ValueError:
        raise ValueError("Output path must remain inside case2_design: %s" % resolved)
    return resolved


def _task_id(spec):  # type: (RunSpec) -> str
    return "%s/seed_%d" % (spec.method_id, spec.seed)


def _diagnostic_templates():  # type: () -> List[Dict[str, Any]]
    defaults = ActionParameters()
    templates = []
    for unavailable in ("A2", "A3", "A4"):
        templates.append(
            {
                "method": "LLM-EP",
                "variant": "without_%s" % unavailable.lower(),
                "parameters": defaults,
                "selector_options": {"unavailable_actions": [unavailable]},
            }
        )

    templates.append(
        {"method": "A2-only", "variant": "default", "parameters": defaults}
    )
    for value in (0.3, 0.7):
        templates.append(
            {
                "method": "A2-only",
                "variant": "f_%s" % str(value).replace(".", "p"),
                "parameters": replace(defaults, a2_differential_weight=value),
            }
        )
    templates.append(
        {
            "method": "A2-only",
            "variant": "cr_0p7",
            "parameters": replace(defaults, a2_crossover_probability=0.7),
        }
    )
    for value in (0.20, 0.30):
        templates.append(
            {
                "method": "A2-only",
                "variant": "target_%s" % str(value).replace(".", "p"),
                "parameters": replace(defaults, a2_target_fraction=value),
            }
        )

    templates.append(
        {"method": "A3-only", "variant": "default", "parameters": defaults}
    )
    for value in (0.02, 0.10):
        templates.append(
            {
                "method": "A3-only",
                "variant": "sigma0_%s" % str(value).replace(".", "p"),
                "parameters": replace(defaults, a3_sigma_0=value),
            }
        )
    for value in (0.10, 0.20):
        templates.append(
            {
                "method": "A3-only",
                "variant": "target_%s" % str(value).replace(".", "p"),
                "parameters": replace(defaults, a3_target_fraction=value),
            }
        )

    for interval in (10, 20, 40):
        templates.append(
            {
                "method": "Rule",
                "variant": "interval_%d" % interval,
                "parameters": defaults,
                "search_config": replace(SearchConfig(), decision_interval=interval),
            }
        )
    return templates


def build_run_specs(
    group,
    experiment_id,
    methods=None,
    seeds=None,
    llm_options=None,
):
    # type: (str, str, Optional[Sequence[str]], Optional[Sequence[int]], Optional[Dict[str, Any]]) -> List[RunSpec]
    if methods and "non-llm" in methods:
        if tuple(methods) != ("non-llm",):
            raise ValueError("non-llm cannot be combined with explicit method names")
        methods = NON_LLM_METHODS

    if group == "baseline":
        default_methods, default_seeds = ("A1-only",), BASELINE_SEEDS
    elif group == "validation":
        default_methods, default_seeds = FORMAL_METHODS, VALIDATION_SEEDS
    elif group == "formal":
        default_methods, default_seeds = FORMAL_METHODS, FORMAL_SEEDS
    elif group == "diagnostic":
        chosen_seeds = tuple(seeds or DIAGNOSTIC_SEEDS)
        method_filter = set(methods) if methods else None
        unknown = sorted((method_filter or set()) - set(FORMAL_METHODS))
        if unknown:
            raise ValueError("Unknown methods: %s" % ", ".join(unknown))
        specs = []
        for template in _diagnostic_templates():
            if method_filter and template["method"] not in method_filter:
                continue
            for seed in chosen_seeds:
                specs.append(
                    RunSpec(
                        experiment_id=experiment_id,
                        group=group,
                        method=template["method"],
                        variant=template["variant"],
                        seed=seed,
                        search_config=template.get("search_config", SearchConfig()),
                        action_parameters=template["parameters"],
                        selector_options=template.get("selector_options", {}),
                        llm_options=llm_options or {},
                    )
                )
        if not specs:
            raise ValueError("No diagnostic variants match the requested methods")
        return specs
    else:
        raise ValueError("Unknown experiment group: %s" % group)

    chosen_methods = tuple(methods or default_methods)
    unknown = sorted(set(chosen_methods) - set(FORMAL_METHODS))
    if unknown:
        raise ValueError("Unknown methods: %s" % ", ".join(unknown))
    chosen_seeds = tuple(seeds or default_seeds)
    return [
        RunSpec(
            experiment_id=experiment_id,
            group=group,
            method=method,
            seed=seed,
            llm_options=llm_options or {},
        )
        for method in chosen_methods
        for seed in chosen_seeds
    ]


def _create_selector(spec, llm_provider=None):
    try:
        from .strategy_selectors import create_selector
    except ImportError:
        from strategy_selectors import create_selector
    return create_selector(spec, llm_provider=llm_provider)


def _outcome_rows(outcomes):
    return [to_jsonable(outcome) for outcome in outcomes]


def _checkpoint_costs(history, budget):
    result = {}
    for fraction in (0.25, 0.50, 0.75, 1.00):
        target = fraction * budget
        eligible = [row for row in history if row["search_nfe"] <= target]
        row = eligible[-1] if eligible else history[0]
        result[str(int(fraction * 100))] = {
            "search_nfe": int(row["search_nfe"]),
            "best_cost": _finite_float(row["best_cost"]),
            "best_cv": _finite_float(row["best_cv"]),
            "feasible": bool(row["feasible"]),
        }
    return result


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _result_scalar(result, key):
    try:
        return _finite_float(result[key][0])
    except (KeyError, IndexError, TypeError):
        return None


def _result_is_finite(result):
    try:
        arrays = [value for value in result.values() if isinstance(value, np.ndarray)]
        return bool(arrays) and all(np.isfinite(value).all() for value in arrays)
    except (TypeError, ValueError):
        return False


def _sanitize_metrics(metrics):
    return {key: _finite_float(value) for key, value in metrics.items()}


def _llm_records(selector):
    for name in ("llm_calls", "call_records"):
        records = getattr(selector, name, None)
        if records is not None:
            return [to_jsonable(record) for record in records]
    return []


def run_one(
    spec,
    results_root=DEFAULT_RESULTS_ROOT,
    resume=False,
    solver_fn=None,
    llm_provider=None,
):
    root = _require_case2_path(Path(results_root))
    paths = RunPaths.from_spec(root, spec)
    _require_case2_path(paths.directory)
    ensure_run_config(paths, spec)
    previous = completed_summary(paths)
    if previous is not None:
        return {"state": "skipped_complete", "path": str(paths.directory), "summary": previous}
    existing_outputs = any(
        paths.file(name).exists()
        for name in (
            "history.csv",
            "actions.jsonl",
            "schedule.csv",
            "validation.json",
            "attempts.jsonl",
        )
    )
    if existing_outputs and not resume:
        raise RuntimeError(
            "Incomplete run exists; pass --resume to rerun the whole run: %s"
            % paths.directory
        )

    started_at = _utc_now()
    started = time.perf_counter()
    try:
        try:
            from . import solve_case2 as solver
        except ImportError:
            import solve_case2 as solver
        selector = _create_selector(spec, llm_provider=llm_provider)
        hours, p_vital, p_nonvital, p_pv = solver.load_input_data(BASE_DIR.parent / "data.md")
        search = solver_fn or solver.solve_experiment
        run = search(
            p_vital,
            p_nonvital,
            p_pv,
            seed=spec.seed,
            selector=selector,
            config=spec.search_config,
            action_parameters=spec.action_parameters,
        )
        elapsed = time.perf_counter() - started
        result = run.audit_evaluation
        validation_error = None
        try:
            metrics = solver.validation_metrics(result, p_vital, p_nonvital, p_pv)
            failures = solver.validation_failures(
                result,
                metrics,
                feasibility_tolerance=spec.search_config.feasibility_tolerance,
            )
        except Exception as error:
            metrics = {}
            validation_error = "%s: %s" % (type(error).__name__, error)
            failures = ["numerical validation failed: %s" % validation_error]
        budget_ok = run.search_nfe == spec.search_config.search_budget
        if not budget_ok:
            failures.append(
                "search NFE is %d, expected %d"
                % (run.search_nfe, spec.search_config.search_budget)
            )
        finite = _result_is_finite(result)
        feasible = not failures
        if feasible:
            outcome = "feasible"
        elif validation_error is not None or not finite or not budget_ok:
            outcome = "numerical_failure"
        else:
            outcome = "no_feasible"

        validation = {
            "valid": feasible,
            "failures": failures,
            "metrics": _sanitize_metrics(metrics),
            "search_nfe_valid": budget_ok,
            "validation_error": validation_error,
        }
        action_rows = _outcome_rows(run.outcomes)
        action_counts = {}
        for row in action_rows:
            action = row["applied_action"]
            action_counts[action] = action_counts.get(action, 0) + 1
        llm_calls = _llm_records(selector) if spec.method.startswith("LLM-") else None
        summary = {
            "status": "complete",
            "outcome": outcome,
            "experiment_id": spec.experiment_id,
            "group": spec.group,
            "method": spec.method,
            "method_id": spec.method_id,
            "variant": spec.variant,
            "seed": spec.seed,
            "search_nfe": int(run.search_nfe),
            "audit_nfe": int(run.audit_nfe),
            "feasible": feasible,
            "best_cost": _result_scalar(result, "total_cost"),
            "best_cv": _result_scalar(result, "cv"),
            "cv_r": _result_scalar(result, "cv_r"),
            "cv_soc": _result_scalar(result, "cv_soc"),
            "cv_e": _result_scalar(result, "cv_e"),
            "cv_d": _result_scalar(result, "cv_d"),
            "total_emissions": _result_scalar(result, "total_emissions"),
            "first_feasible_nfe": run.first_feasible_nfe,
            "budget_checkpoints": _checkpoint_costs(run.history, spec.search_config.search_budget),
            "wall_seconds": elapsed,
            "selector_seconds": float(sum(row["elapsed_seconds"] for row in action_rows)),
            "action_counts": action_counts,
            "selector_fallbacks": int(sum(bool(row["fallback_used"]) for row in action_rows)),
            "llm_call_count": len(llm_calls or []),
            "validation_failures": failures,
            "started_at": started_at,
            "completed_at": _utc_now(),
        }

        def schedule_writer(path):
            if validation_error is None:
                solver.write_schedule(path, hours, p_vital, p_nonvital, p_pv, result)
            else:
                Path(path).write_text(
                    "status,error\nnumerical_failure,%s\n"
                    % validation_error.replace("\n", " ").replace(",", ";"),
                    encoding="utf-8",
                )

        commit_run_artifacts(
            paths,
            history=run.history,
            history_fields=HISTORY_FIELDS,
            actions=action_rows,
            schedule_writer=schedule_writer,
            validation=validation,
            summary=summary,
            llm_calls=llm_calls,
        )
        return {"state": "complete", "path": str(paths.directory), "summary": summary}
    except Exception as error:
        append_attempt(
            paths,
            {
                "at": _utc_now(),
                "state": "infrastructure_or_configuration_error",
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def experiment_status(experiment_id, results_root=DEFAULT_RESULTS_ROOT):
    results_root = _require_case2_path(Path(results_root))
    root = _require_case2_path(results_root / experiment_id)
    rows = []
    if not root.exists():
        return rows
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        return [
            {
                "method_id": task["method_id"],
                "seed": task["seed"],
                "state": task["status"],
                "path": str(root / task["relative_path"]),
            }
            for task in manifest["tasks"]
        ]
    for config_path in sorted(root.glob("*/seed_*/config.json")):
        directory = config_path.parent
        config = read_json(config_path)
        summary_path = directory / "summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            state = summary.get("outcome", "complete")
        elif (directory / "attempts.jsonl").exists():
            state = "infrastructure_incomplete"
        else:
            state = "not_started_or_interrupted"
        rows.append(
            {
                "method_id": config["method_id"],
                "seed": config["seed"],
                "state": state,
                "path": str(directory),
            }
        )
    return rows


def _parse_csv_values(text):
    return tuple(value.strip() for value in text.split(",") if value.strip())


def _parse_seeds(text):
    if not text:
        return None
    seeds = []
    for item in _parse_csv_values(text):
        if ".." in item:
            start, end = (int(value) for value in item.split("..", 1))
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(item))
    return tuple(seeds)


def _load_llm_options(path):
    return read_json(Path(path)) if path else {}


def _run_command(args, llm_provider=None):
    methods = None if args.methods == "all" else _parse_csv_values(args.methods)
    specs = build_run_specs(
        args.group,
        args.experiment_id,
        methods=methods,
        seeds=_parse_seeds(args.seeds),
        llm_options=_load_llm_options(args.llm_config),
    )
    results_root = _require_case2_path(Path(args.results_root))
    for spec in specs:
        _require_case2_path(RunPaths.from_spec(results_root, spec).directory)
    uses_llm = any(spec.method.startswith("LLM-") for spec in specs)
    if uses_llm and llm_provider is None:
        raise LLMConfigurationError(
            "LLM experiments require an injected provider; the CLI does not bundle one"
        )
    if uses_llm and args.jobs != 1:
        raise LLMConfigurationError(
            "Injected LLM providers are supported only with --jobs 1"
        )

    manifest_path = initialize_experiment_manifest(
        results_root,
        specs,
        timestamp=_utc_now(),
        resume=args.resume,
    )
    failures = 0
    if args.jobs == 1:
        for spec in specs:
            task_id = _task_id(spec)
            update_manifest_task(manifest_path, task_id, "running", _utc_now())
            try:
                result = run_one(
                    spec,
                    results_root,
                    args.resume,
                    llm_provider=llm_provider,
                )
                summary = result.get("summary", {})
                update_manifest_task(
                    manifest_path,
                    task_id,
                    result["state"],
                    _utc_now(),
                    outcome=summary.get("outcome"),
                )
                print("%s seed=%d %s" % (spec.method_id, spec.seed, result["state"]))
            except Exception as error:
                failures += 1
                update_manifest_task(
                    manifest_path,
                    task_id,
                    "failed",
                    _utc_now(),
                    error="%s: %s" % (type(error).__name__, error),
                )
                print("%s seed=%d ERROR: %s" % (spec.method_id, spec.seed, error), file=sys.stderr)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {}
            for spec in specs:
                update_manifest_task(
                    manifest_path, _task_id(spec), "running", _utc_now()
                )
                future = executor.submit(run_one, spec, results_root, args.resume)
                futures[future] = spec
            for future in concurrent.futures.as_completed(futures):
                spec = futures[future]
                task_id = _task_id(spec)
                try:
                    result = future.result()
                    summary = result.get("summary", {})
                    update_manifest_task(
                        manifest_path,
                        task_id,
                        result["state"],
                        _utc_now(),
                        outcome=summary.get("outcome"),
                    )
                    print("%s seed=%d %s" % (spec.method_id, spec.seed, result["state"]))
                except Exception as error:
                    failures += 1
                    update_manifest_task(
                        manifest_path,
                        task_id,
                        "failed",
                        _utc_now(),
                        error="%s: %s" % (type(error).__name__, error),
                    )
                    print("%s seed=%d ERROR: %s" % (spec.method_id, spec.seed, error), file=sys.stderr)
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run and inspect Case 2 experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--experiment-id", required=True)
    run_parser.add_argument("--group", choices=("baseline", "validation", "formal", "diagnostic"), required=True)
    run_parser.add_argument("--methods", default="all")
    run_parser.add_argument("--seeds", help="Comma-separated seeds or inclusive ranges such as 1..3")
    run_parser.add_argument("--jobs", type=int, default=1)
    run_parser.add_argument("--resume", action="store_true", help="Rerun incomplete runs from initialization")
    run_parser.add_argument("--llm-config")
    run_parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--experiment-id", required=True)
    status_parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.jobs < 1:
            parser.error("--jobs must be at least 1")
        return _run_command(args)
    rows = experiment_status(args.experiment_id, Path(args.results_root))
    for row in rows:
        print("{method_id}\tseed={seed}\t{state}\t{path}".format(**row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
