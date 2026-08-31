"""Oracle action branch experiment for fixed UniformRandom checkpoints."""

from dataclasses import asdict
from pathlib import Path
import argparse
import copy
import csv
import json
import math
import re
import sys

import numpy as np

try:
    from . import solve_case2 as solver
    from .experiment_io import atomic_write_csv, atomic_write_json, atomic_write_text, read_json, to_jsonable
    from .experiment_types import Action, ActionParameters, RandomStreams, SearchConfig, SearchState
    from .strategy_selectors import UniformRandomSelector
except ImportError:
    import solve_case2 as solver
    from experiment_io import atomic_write_csv, atomic_write_json, atomic_write_text, read_json, to_jsonable
    from experiment_types import Action, ActionParameters, RandomStreams, SearchConfig, SearchState
    from strategy_selectors import UniformRandomSelector


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = BASE_DIR / "results"
DEFAULT_UNIFORM_EXPERIMENT = "uniform_random_formal"
DEFAULT_ORACLE_EXPERIMENT = "oracle_action_branch"
GUROBI_REFERENCE_COST = 42356.702428238088
CHECKPOINT_FRACTIONS = (0.20, 0.40, 0.60, 0.80, 0.90)
ACTIONS = (Action.A1, Action.A2, Action.A3, Action.A4)
ACTION_CODES = {Action.A1: 1, Action.A2: 2, Action.A3: 3, Action.A4: 4}
CLEAR_ORACLE_STD_MULTIPLIER = 2.0


def _require_case2_path(path):
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(BASE_DIR)
    except ValueError as error:
        raise ValueError("Output path must remain inside case2_design: %s" % resolved) from error
    return resolved


def _checkpoint_label(fraction):
    return "%02dpct" % int(round(100.0 * fraction))


def _checkpoint_targets(config):
    budget = config.search_budget
    targets = []
    for fraction in CHECKPOINT_FRACTIONS:
        target_nfe = fraction * budget
        targets.append(
            {
                "fraction": fraction,
                "target_nfe": int(round(target_nfe)),
                "label": _checkpoint_label(fraction),
            }
        )
    return targets


def select_seeds(uniform_results_root, count=5):
    """Select deterministic nearest-rank cost quantiles from UniformRandom."""
    root = Path(uniform_results_root)
    summaries = []
    for path in sorted(root.glob("UniformRandom/seed_*/summary.json")):
        summary = read_json(path)
        if summary.get("status") == "complete":
            summaries.append(summary)
    if len(summaries) < count:
        raise ValueError("UniformRandom results contain fewer than %d complete seeds" % count)
    summaries.sort(key=lambda item: (float(item["best_cost"]), int(item["seed"])))
    if count == 1:
        indices = [0]
    else:
        # Nearest-rank quantiles: ranks ceil(q*n), converted to zero-based indices.
        indices = [
            0 if i == 0 else min(
                len(summaries) - 1,
                int(math.ceil(i * len(summaries) / (count - 1))) - 1,
            )
            for i in range(count)
        ]
    return [summaries[index] for index in indices]


def _rng_from_state(state):
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(state)
    return rng


def _state_arrays(state):
    arrays = {
        "positions": state.positions,
        "velocities": state.velocities,
        "previous_positions": state.previous_positions,
        "previous_velocities": state.previous_velocities,
        "personal_best_positions": state.personal_best_positions,
        "personal_best_cost": state.personal_best_cost,
        "personal_best_cv": state.personal_best_cv,
        "personal_best_cv_components": state.personal_best_cv_components,
        "global_best_position": state.global_best_position,
    }
    arrays.update({"evaluation__" + key: value for key, value in state.evaluation.items()})
    return arrays


def _state_metadata(state, streams, seed, checkpoint):
    return {
        "schema_version": 1,
        "seed": int(seed),
        "checkpoint": checkpoint,
        "iteration": int(state.iteration),
        "search_nfe": int(state.search_nfe),
        "global_best_cost": float(state.global_best_cost),
        "global_best_cv": float(state.global_best_cv),
        "global_best_cv_components": list(state.global_best_cv_components),
        "first_feasible_nfe": state.first_feasible_nfe,
        "active_action": state.active_action.value if state.active_action is not None else None,
        "stagnation_iterations": int(state.stagnation_iterations),
        "evaluation_keys": sorted(state.evaluation.keys()),
        "rng_states": {
            "core": to_jsonable(streams.core.bit_generator.state),
            "a2": to_jsonable(streams.a2.bit_generator.state),
            "a3": to_jsonable(streams.a3.bit_generator.state),
            "a4": to_jsonable(streams.a4.bit_generator.state),
            "selector": to_jsonable(streams.selector.bit_generator.state),
        },
    }


def save_checkpoint(directory, state, streams, seed, checkpoint):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "state.npz", **_state_arrays(state))
    atomic_write_json(directory / "state.json", _state_metadata(state, streams, seed, checkpoint))


def load_checkpoint(directory):
    directory = Path(directory)
    metadata = read_json(directory / "state.json")
    with np.load(directory / "state.npz", allow_pickle=False) as data:
        arrays = {key: np.array(data[key], copy=True) for key in data.files}
    evaluation = {
        key: arrays["evaluation__" + key]
        for key in metadata["evaluation_keys"]
    }
    state = SearchState(
        iteration=int(metadata["iteration"]),
        search_nfe=int(metadata["search_nfe"]),
        positions=arrays["positions"],
        velocities=arrays["velocities"],
        previous_positions=arrays["previous_positions"],
        previous_velocities=arrays["previous_velocities"],
        evaluation=evaluation,
        personal_best_positions=arrays["personal_best_positions"],
        personal_best_cost=arrays["personal_best_cost"],
        personal_best_cv=arrays["personal_best_cv"],
        personal_best_cv_components=arrays["personal_best_cv_components"],
        global_best_position=arrays["global_best_position"],
        global_best_cost=float(metadata["global_best_cost"]),
        global_best_cv=float(metadata["global_best_cv"]),
        global_best_cv_components=tuple(metadata["global_best_cv_components"]),
        first_feasible_nfe=metadata["first_feasible_nfe"],
        active_action=(Action(metadata["active_action"]) if metadata["active_action"] else None),
        stagnation_iterations=int(metadata["stagnation_iterations"]),
    )
    rng_states = metadata["rng_states"]
    streams = RandomStreams(
        core=_rng_from_state(rng_states["core"]),
        a2=_rng_from_state(rng_states["a2"]),
        a3=_rng_from_state(rng_states["a3"]),
        a4=_rng_from_state(rng_states["a4"]),
        selector=_rng_from_state(rng_states["selector"]),
    )
    return state, streams, metadata


def _branch_streams(checkpoint_streams, seed, checkpoint_iteration, repeat, action):
    """Use the checkpoint core stream and independent action streams."""
    return RandomStreams(
        core=_rng_from_state(checkpoint_streams.core.bit_generator.state),
        a2=np.random.default_rng(np.random.SeedSequence([seed, checkpoint_iteration, repeat, ACTION_CODES[action], 102])),
        a3=np.random.default_rng(np.random.SeedSequence([seed, checkpoint_iteration, repeat, ACTION_CODES[action], 103])),
        a4=np.random.default_rng(np.random.SeedSequence([seed, checkpoint_iteration, repeat, ACTION_CODES[action], 104])),
        selector=_rng_from_state(checkpoint_streams.selector.bit_generator.state),
    )


def run_action_window(
    state,
    streams,
    action,
    p_vital,
    p_nonvital,
    p_pv,
    config,
    action_parameters,
    iterations=20,
):
    before_cost = float(state.global_best_cost)
    before_cv = float(state.global_best_cv)
    for _ in range(iterations):
        x_base, v_base = solver.mppso_step(state, streams.core, config)
        x_action = solver.apply_action(action, x_base, state, streams, action_parameters, config)
        positions = solver.canonicalize(x_action, config)
        evaluation = solver.evaluate(
            positions, p_vital, p_nonvital, p_pv, penalty_lambda=config.penalty_lambda
        )
        state.search_nfe += config.population_size
        state.iteration += 1
        solver.advance_search_state(state, positions, v_base, evaluation, config)
    after_cost = float(state.global_best_cost)
    after_cv = float(state.global_best_cv)
    return {
        "start_iteration": int(state.iteration - iterations),
        "end_iteration": int(state.iteration),
        "start_nfe": int(state.search_nfe - iterations * config.population_size),
        "end_nfe": int(state.search_nfe),
        "before_cost": before_cost,
        "after_cost": after_cost,
        "before_cv": before_cv,
        "after_cv": after_cv,
        "feasible_after": bool(after_cv <= config.feasibility_tolerance),
        "cost_reward": before_cost - after_cost,
        "gap_reward": (before_cost - GUROBI_REFERENCE_COST) / GUROBI_REFERENCE_COST
        - (after_cost - GUROBI_REFERENCE_COST) / GUROBI_REFERENCE_COST,
    }


def _summary_stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _rank_actions(median_rewards):
    ordered = sorted(ACTIONS, key=lambda action: (-median_rewards[action], ACTION_CODES[action]))
    return ordered, ">".join(action.value for action in ordered)


def _oracle_metrics(
    action_rewards,
    action_stds,
    action_gap_rewards,
    clear_multiplier=CLEAR_ORACLE_STD_MULTIPLIER,
):
    ordered, rank_string = _rank_actions(action_rewards)
    best, second = ordered[:2]
    oracle_reward = float(action_rewards[best])
    random_reward = float(np.mean([action_rewards[action] for action in ACTIONS]))
    margin = oracle_reward - float(action_rewards[second])
    pooled_std = float(
        np.sqrt(np.mean([float(action_stds[action]) ** 2 for action in ACTIONS]))
    )
    if pooled_std > 0.0:
        margin_noise_ratio = margin / pooled_std
    else:
        margin_noise_ratio = float("inf") if margin > 0.0 else 0.0
    oracle_gap_reward = float(max(action_gap_rewards.values()))
    random_gap_reward = float(
        np.mean([action_gap_rewards[action] for action in ACTIONS])
    )
    clear = bool(margin > clear_multiplier * pooled_std)
    return {
        "oracle_action": best.value,
        "second_best_action": second.value,
        "action_rank": rank_string,
        "oracle_reward": oracle_reward,
        "random_reward_mean": random_reward,
        "delta_reward": oracle_reward - random_reward,
        "relative_delta_vs_random": (
            (oracle_reward - random_reward) / random_reward
            if random_reward != 0.0
            else float("nan")
        ),
        "margin": margin,
        "pooled_repeat_std": pooled_std,
        "margin_noise_ratio": margin_noise_ratio,
        "oracle_class": "Clear Oracle" if clear else "Ambiguous Oracle",
        "oracle_gap_reward": oracle_gap_reward,
        "random_gap_reward": random_gap_reward,
        "delta_gap_reward": oracle_gap_reward - random_gap_reward,
    }


def _read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _phase_summary_rows(enriched_rows):
    rows = []
    for checkpoint in (_checkpoint_label(value) for value in CHECKPOINT_FRACTIONS):
        group = [row for row in enriched_rows if row["checkpoint"] == checkpoint]
        if not group:
            continue
        result = {"checkpoint": checkpoint, "checkpoint_count": len(group)}
        for field in (
            "oracle_reward",
            "random_reward_mean",
            "delta_reward",
            "relative_delta_vs_random",
            "margin",
            "pooled_repeat_std",
            "margin_noise_ratio",
            "delta_gap_reward",
        ):
            values = np.asarray([float(row[field]) for row in group], dtype=float)
            result.update(
                {
                    field + "_median": float(np.median(values)),
                    field + "_mean": float(np.mean(values)),
                    field + "_q25": float(np.quantile(values, 0.25)),
                    field + "_q75": float(np.quantile(values, 0.75)),
                }
            )
        clear_count = sum(row["oracle_class"] == "Clear Oracle" for row in group)
        result["clear_count"] = clear_count
        result["ambiguous_count"] = len(group) - clear_count
        result["clear_fraction"] = clear_count / len(group)
        for action in ACTIONS:
            result[action.value + "_oracle_count"] = sum(
                row["oracle_action"] == action.value for row in group
            )
        rows.append(result)
    return rows


def _write_oracle_report(path, enriched_rows, phase_rows):
    clear_count = sum(row["oracle_class"] == "Clear Oracle" for row in enriched_rows)
    total = len(enriched_rows)
    delta = np.asarray([float(row["delta_reward"]) for row in enriched_rows])
    margin = np.asarray([float(row["margin"]) for row in enriched_rows])
    pooled = np.asarray([float(row["pooled_repeat_std"]) for row in enriched_rows])
    lines = [
        "# Oracle vs Uniform Random Action Reward",
        "",
        "## Definition",
        "",
        "- Each action reward is the median of its three branch repeats.",
        "- `R_oracle` is the largest of the four action median rewards.",
        "- `R_random` is the uniform average of the four action median rewards.",
        "- `Delta R = R_oracle - R_random`.",
        "- `Margin` is the best minus second-best action median reward.",
        "- Clear Oracle requires `Margin > %.1f x pooled_repeat_std`; otherwise it is Ambiguous Oracle."
        % CLEAR_ORACLE_STD_MULTIPLIER,
        "",
        "## Overall",
        "",
        "- Checkpoints: `%d`" % total,
        "- Clear Oracle: `%d/%d`" % (clear_count, total),
        "- Ambiguous Oracle: `%d/%d`" % (total - clear_count, total),
        "- Median Delta R: `%.6f`" % float(np.median(delta)),
        "- Median Margin: `%.6f`" % float(np.median(margin)),
        "- Median pooled repeat standard deviation: `%.6f`" % float(np.median(pooled)),
        "",
        "## By Checkpoint",
        "",
        "| Checkpoint | Oracle median | Random median | Delta R median | Relative uplift median | Margin median | Clear | Ambiguous |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in phase_rows:
        lines.append(
            "| %s | %.6f | %.6f | %.6f | %.2f%% | %.6f | %d | %d |"
            % (
                row["checkpoint"],
                row["oracle_reward_median"],
                row["random_reward_mean_median"],
                row["delta_reward_median"],
                100.0 * row["relative_delta_vs_random_median"],
                row["margin_median"],
                row["clear_count"],
                row["ambiguous_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "`R_random` is a uniform-action proxy computed from four action medians, not a new Monte Carlo estimate.",
            "With only three repeats per action, Clear/Ambiguous is a descriptive noise screen rather than a significance test.",
            "The hindsight Oracle is an upper-bound diagnostic and is not an online deployable selector.",
            "",
        ]
    )
    atomic_write_text(path, "\n".join(lines))


def analyze_oracle_results(output_root):
    output_root = _require_case2_path(output_root)
    summary_path = output_root / "checkpoint_summary.csv"
    repeats_path = output_root / "action_repeats.csv"
    summary_rows = _read_csv_rows(summary_path)
    repeat_rows = _read_csv_rows(repeats_path)
    grouped = {}
    for row in repeat_rows:
        key = (int(row["seed"]), row["checkpoint"], Action(row["action"]))
        grouped.setdefault(key, []).append(row)

    enriched_rows = []
    for row in summary_rows:
        seed = int(row["seed"])
        checkpoint = row["checkpoint"]
        action_rewards = {}
        action_stds = {}
        action_gap_rewards = {}
        for action in ACTIONS:
            repeats = grouped.get((seed, checkpoint, action), [])
            if not repeats:
                raise ValueError(
                    "Missing repeats for seed %d, checkpoint %s, action %s"
                    % (seed, checkpoint, action.value)
                )
            rewards = [float(item["cost_reward"]) for item in repeats]
            gaps = [float(item["gap_reward"]) for item in repeats]
            action_rewards[action] = float(np.median(rewards))
            action_stds[action] = (
                float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
            )
            action_gap_rewards[action] = float(np.median(gaps))
        enriched = dict(row)
        enriched.update(
            _oracle_metrics(action_rewards, action_stds, action_gap_rewards)
        )
        enriched_rows.append(enriched)

    summary_fields = list(enriched_rows[0].keys()) if enriched_rows else []
    atomic_write_csv(summary_path, summary_fields, enriched_rows)
    phase_rows = _phase_summary_rows(enriched_rows)
    phase_path = output_root / "oracle_vs_random_by_checkpoint.csv"
    phase_fields = list(phase_rows[0].keys()) if phase_rows else []
    atomic_write_csv(phase_path, phase_fields, phase_rows)
    report_path = output_root / "oracle_vs_random_report.md"
    _write_oracle_report(report_path, enriched_rows, phase_rows)
    result = {
        "checkpoint_summary": summary_path,
        "phase_summary": phase_path,
        "report": report_path,
        "checkpoint_count": len(enriched_rows),
        "clear_count": sum(
            row["oracle_class"] == "Clear Oracle" for row in enriched_rows
        ),
    }
    run_summary_path = output_root / "run_summary.json"
    if run_summary_path.exists():
        run_summary = read_json(run_summary_path)
        run_summary.update(
            {
                "oracle_vs_random_by_checkpoint": str(phase_path),
                "oracle_vs_random_report": str(report_path),
                "clear_oracle_count": result["clear_count"],
                "ambiguous_oracle_count": len(enriched_rows) - result["clear_count"],
            }
        )
        atomic_write_json(run_summary_path, run_summary)
    return result


def run_oracle_experiment(
    uniform_results_root=DEFAULT_RESULTS_ROOT,
    output_root=None,
    seed_count=5,
    repeats=3,
    window_iterations=20,
):
    uniform_root = Path(uniform_results_root) / DEFAULT_UNIFORM_EXPERIMENT
    output_root = _require_case2_path(output_root or (DEFAULT_RESULTS_ROOT / DEFAULT_ORACLE_EXPERIMENT))
    selected = select_seeds(uniform_root, seed_count)
    _, p_vital, p_nonvital, p_pv = solver.load_input_data(BASE_DIR.parent / "data.md")
    config = solver.default_search_config()
    action_parameters = ActionParameters()
    targets = _checkpoint_targets(config)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "experiment": DEFAULT_ORACLE_EXPERIMENT,
        "uniform_experiment": DEFAULT_UNIFORM_EXPERIMENT,
        "selected_seeds": [int(item["seed"]) for item in selected],
        "selection_rule": "final_cost_sorted_nearest_ranks_1_8_15_23_30",
        "checkpoint_fractions": list(CHECKPOINT_FRACTIONS),
        "checkpoint_rounding": "first_search_nfe_at_or_above_target",
        "branch_iterations": int(window_iterations),
        "repeats_per_action": int(repeats),
        "rng_policy": "common_core_per_repeat_and_independent_action_streams",
        "gurobi_reference_cost": GUROBI_REFERENCE_COST,
        "expected_checkpoint_count": len(selected) * len(targets),
        "expected_branch_count": len(selected) * len(targets) * len(ACTIONS) * repeats,
        "selected_uniform_summaries": selected,
    }
    atomic_write_json(output_root / "manifest.json", manifest)

    repeat_rows = []
    summary_rows = []
    for selected_summary in selected:
        seed = int(selected_summary["seed"])
        seed_root = output_root / ("seed_%d" % seed)
        checkpoint_dirs = {}
        target_index = 0

        def checkpoint_callback(state, streams, row):
            nonlocal target_index
            if target_index >= len(targets):
                return
            target = targets[target_index]
            if state.search_nfe < target["target_nfe"]:
                return
            checkpoint = dict(target)
            checkpoint["actual_nfe"] = int(state.search_nfe)
            checkpoint["actual_fraction"] = float(state.search_nfe / config.search_budget)
            checkpoint["iteration"] = int(state.iteration)
            directory = seed_root / "checkpoints" / checkpoint["label"]
            save_checkpoint(directory, state, streams, seed, checkpoint)
            checkpoint_dirs[checkpoint["label"]] = directory
            target_index += 1

        selector = UniformRandomSelector()
        baseline = solver.solve_experiment(
            p_vital,
            p_nonvital,
            p_pv,
            seed=seed,
            selector=selector,
            config=config,
            action_parameters=action_parameters,
            checkpoint_callback=checkpoint_callback,
        )
        if target_index != len(targets):
            raise RuntimeError("failed to capture all checkpoints for seed %d" % seed)
        if not np.isclose(float(baseline.audit_evaluation["total_cost"][0]), float(selected_summary["best_cost"]), atol=1e-8):
            raise RuntimeError("UniformRandom replay mismatch for seed %d" % seed)
        atomic_write_json(
            seed_root / "baseline_replay.json",
            {
                "seed": seed,
                "best_cost": float(baseline.audit_evaluation["total_cost"][0]),
                "best_cv": float(baseline.audit_evaluation["cv"][0]),
                "search_nfe": int(baseline.search_nfe),
                "history_rows": len(baseline.history),
                "checkpoint_labels": list(checkpoint_dirs),
            },
        )

        for target in targets:
            directory = checkpoint_dirs[target["label"]]
            checkpoint_state, checkpoint_streams, checkpoint_metadata = load_checkpoint(directory)
            before_cost = float(checkpoint_state.global_best_cost)
            action_stats = {}
            for action in ACTIONS:
                branch_results = []
                for repeat in range(repeats):
                    state, _, _ = load_checkpoint(directory)
                    streams = _branch_streams(
                        checkpoint_streams,
                        seed,
                        int(checkpoint_metadata["iteration"]),
                        repeat,
                        action,
                    )
                    result = run_action_window(
                        state,
                        streams,
                        action,
                        p_vital,
                        p_nonvital,
                        p_pv,
                        config,
                        action_parameters,
                        iterations=window_iterations,
                    )
                    result.update(
                        {
                            "seed": seed,
                            "checkpoint": target["label"],
                            "action": action.value,
                            "repeat": repeat + 1,
                            "checkpoint_nfe": int(checkpoint_metadata["search_nfe"]),
                        }
                    )
                    repeat_rows.append(result)
                    branch_results.append(result)
                action_stats[action] = branch_results

            median_rewards = {
                action: float(np.median([item["cost_reward"] for item in action_stats[action]]))
                for action in ACTIONS
            }
            median_gap_rewards = {
                action: float(np.median([item["gap_reward"] for item in action_stats[action]]))
                for action in ACTIONS
            }
            ordered, rank_string = _rank_actions(median_rewards)
            row = {
                "seed": seed,
                "checkpoint": target["label"],
                "target_fraction": target["fraction"],
                "target_nfe": target["target_nfe"],
                "actual_fraction": checkpoint_metadata["checkpoint"]["actual_fraction"],
                "actual_nfe": checkpoint_metadata["search_nfe"],
                "iteration": checkpoint_metadata["iteration"],
                "before_cost": before_cost,
                "before_cv": checkpoint_metadata["global_best_cv"],
                "oracle_action": ordered[0].value,
                "action_rank": rank_string,
            }
            for action in ACTIONS:
                values = [item["cost_reward"] for item in action_stats[action]]
                gaps = [item["gap_reward"] for item in action_stats[action]]
                stats = _summary_stats(values)
                gap_stats = _summary_stats(gaps)
                prefix = action.value
                row.update(
                    {
                        prefix + "_reward_median": stats["median"],
                        prefix + "_reward_mean": stats["mean"],
                        prefix + "_reward_std": stats["std"],
                        prefix + "_reward_q25": stats["q25"],
                        prefix + "_reward_q75": stats["q75"],
                        prefix + "_after_cost_median": float(np.median([item["after_cost"] for item in action_stats[action]])),
                        prefix + "_gap_reward_median": gap_stats["median"],
                    }
                )
            summary_rows.append(row)

    repeat_fields = [
        "seed", "checkpoint", "action", "repeat", "checkpoint_nfe",
        "start_iteration", "end_iteration", "start_nfe", "end_nfe",
        "before_cost", "after_cost", "before_cv", "after_cv",
        "feasible_after", "cost_reward", "gap_reward",
    ]
    atomic_write_csv(output_root / "action_repeats.csv", repeat_fields, repeat_rows)
    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    atomic_write_csv(output_root / "checkpoint_summary.csv", summary_fields, summary_rows)
    analysis = analyze_oracle_results(output_root)
    atomic_write_json(
        output_root / "run_summary.json",
        {
            "manifest": str(output_root / "manifest.json"),
            "checkpoint_summary": str(output_root / "checkpoint_summary.csv"),
            "action_repeats": str(output_root / "action_repeats.csv"),
            "oracle_vs_random_by_checkpoint": str(analysis["phase_summary"]),
            "oracle_vs_random_report": str(analysis["report"]),
            "checkpoint_count": len(summary_rows),
            "branch_count": len(repeat_rows),
            "clear_oracle_count": analysis["clear_count"],
            "ambiguous_oracle_count": len(summary_rows) - analysis["clear_count"],
        },
    )
    return output_root


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Case 2 oracle action branch experiment")
    parser.add_argument("--uniform-results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_RESULTS_ROOT / DEFAULT_ORACLE_EXPERIMENT))
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--window-iterations", type=int, default=20)
    parser.add_argument(
        "--analyze-existing",
        action="store_true",
        help="Recompute Oracle-vs-Random statistics without rerunning branches",
    )
    args = parser.parse_args(argv)
    if args.analyze_existing:
        result = analyze_oracle_results(args.output_root)
        print(result["report"])
        return 0
    if args.seed_count <= 0 or args.repeats <= 0 or args.window_iterations <= 0:
        parser.error("seed-count, repeats, and window-iterations must be positive")
    output = run_oracle_experiment(
        uniform_results_root=args.uniform_results_root,
        output_root=args.output_root,
        seed_count=args.seed_count,
        repeats=args.repeats,
        window_iterations=args.window_iterations,
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
