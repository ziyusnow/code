from pathlib import Path
from typing import Mapping
import argparse
import json
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    from .experiment_io import atomic_write_csv, atomic_write_text, atomic_write_via, read_json
except ImportError:
    from experiment_io import atomic_write_csv, atomic_write_text, atomic_write_via, read_json


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = BASE_DIR / "results"
BOOTSTRAP_SEED = 330001
BOOTSTRAP_RESAMPLES = 10000
TIE_TOLERANCE = 1.0e-6

COMPARISONS = (
    ("A2-only", "A1-only"),
    ("A3-only", "A1-only"),
    ("A4-only", "A1-only"),
    ("UniformRandom", "A1-only"),
    ("Rule", "UniformRandom"),
    ("UCB1", "UniformRandom"),
    ("UCB1", "A1-only"),
    ("LLM-E", "UCB1"),
    ("LLM-EP", "LLM-E"),
    ("LLM-EP", "UCB1"),
)


def _require_case2_path(path):
    resolved = Path(path).resolve()
    base = BASE_DIR.resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError("Analysis output must stay under %s" % base)
    return resolved


def wilson_interval(successes, total, confidence=0.95):
    if total <= 0:
        return (float("nan"), float("nan"))
    if successes < 0 or successes > total:
        raise ValueError("successes must be within [0, total]")
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return (center - radius, center + radius)


def exact_mcnemar(candidate, reference):
    candidate = np.asarray(candidate, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if candidate.shape != reference.shape:
        raise ValueError("paired feasibility arrays must have the same shape")
    candidate_only = int(np.sum(candidate & ~reference))
    reference_only = int(np.sum(~candidate & reference))
    discordant = candidate_only + reference_only
    p_value = (
        1.0
        if discordant == 0
        else float(
            stats.binomtest(
                min(candidate_only, reference_only), discordant, 0.5, alternative="two-sided"
            ).pvalue
        )
    )
    return {
        "candidate_only_feasible": candidate_only,
        "reference_only_feasible": reference_only,
        "discordant": discordant,
        "p_value": p_value,
    }


def rank_biserial(differences):
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values) & (values != 0.0)]
    if values.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(values), method="average")
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    return (positive - negative) / (positive + negative)


def paired_bootstrap_median_ci(
    differences, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED, confidence=0.95
):
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    medians = np.median(values[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    return tuple(float(value) for value in np.quantile(medians, [tail, 1.0 - tail]))


def wilcoxon_signed_rank(differences):
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values) & (values != 0.0)]
    if values.size == 0:
        return {"statistic": 0.0, "p_value": 1.0}
    result = stats.wilcoxon(
        values, zero_method="wilcox", alternative="two-sided", method="auto"
    )
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def holm_adjust(p_values, family_size=None):
    """Adjust a mapping of names to p-values; None remains unavailable."""
    names = list(p_values)
    total = family_size if family_size is not None else len(names)
    if total < len(names):
        raise ValueError("family_size cannot be smaller than the number of entries")
    available = [(name, float(value)) for name, value in p_values.items() if value is not None]
    ordered = sorted(available, key=lambda item: item[1])
    adjusted = {name: None for name in names}
    previous = 0.0
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * value)
        previous = max(previous, candidate)
        adjusted[name] = previous
    return adjusted


def paired_cost_statistics(candidate, reference, minimum_pairs=10):
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape:
        raise ValueError("paired costs must have the same shape")
    valid = np.isfinite(candidate) & np.isfinite(reference)
    candidate, reference = candidate[valid], reference[valid]
    differences = candidate - reference
    relative = np.divide(
        differences,
        reference,
        out=np.full_like(differences, np.nan),
        where=reference != 0.0,
    )
    wins = int(np.sum(differences < -TIE_TOLERANCE))
    ties = int(np.sum(np.abs(differences) <= TIE_TOLERANCE))
    losses = int(np.sum(differences > TIE_TOLERANCE))
    inferential = candidate.size >= minimum_pairs
    absolute_ci = (
        paired_bootstrap_median_ci(differences)
        if inferential
        else (float("nan"), float("nan"))
    )
    relative_ci = (
        paired_bootstrap_median_ci(relative)
        if inferential
        else (float("nan"), float("nan"))
    )
    wilcoxon = wilcoxon_signed_rank(differences) if inferential else None
    return {
        "joint_feasible_pairs": int(candidate.size),
        "candidate_median": float(np.median(candidate)) if candidate.size else float("nan"),
        "candidate_q25": float(np.quantile(candidate, 0.25)) if candidate.size else float("nan"),
        "candidate_q75": float(np.quantile(candidate, 0.75)) if candidate.size else float("nan"),
        "reference_median": float(np.median(reference)) if reference.size else float("nan"),
        "reference_q25": float(np.quantile(reference, 0.25)) if reference.size else float("nan"),
        "reference_q75": float(np.quantile(reference, 0.75)) if reference.size else float("nan"),
        "median_difference": float(np.median(differences)) if differences.size else float("nan"),
        "difference_ci_low": absolute_ci[0],
        "difference_ci_high": absolute_ci[1],
        "median_relative_difference": float(np.median(relative)) if relative.size else float("nan"),
        "relative_ci_low": relative_ci[0],
        "relative_ci_high": relative_ci[1],
        "rank_biserial": rank_biserial(differences),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "cost_inference_available": inferential,
        "cost_analysis_status": "inferential" if inferential else "descriptive_only",
        "wilcoxon_statistic": wilcoxon["statistic"] if wilcoxon else None,
        "wilcoxon_p": wilcoxon["p_value"] if wilcoxon else None,
    }


def step_auc(history, start_nfe, end_nfe, value_column="best_cost"):
    frame = history.sort_values("search_nfe")
    frame = frame[frame["search_nfe"] <= end_nfe]
    if frame.empty or start_nfe >= end_nfe:
        return float("nan")
    interior = frame[
        (frame["search_nfe"] > start_nfe) & (frame["search_nfe"] < end_nfe)
    ]["search_nfe"].astype(int).tolist()
    grid = sorted(set([start_nfe, end_nfe] + interior))
    area = 0.0
    for left, right in zip(grid[:-1], grid[1:]):
        eligible = frame[frame["search_nfe"] <= left]
        if eligible.empty:
            return float("nan")
        value = float(eligible.iloc[-1][value_column])
        area += value * (right - left)
    return area / (end_nfe - start_nfe)


def _prefix_statistics(statistics, prefix):
    return {"%s_%s" % (prefix, key): value for key, value in statistics.items()}


def _paired_auc_statistics(jointly_feasible):
    candidate_auc = []
    reference_auc = []
    for _, pair in jointly_feasible.iterrows():
        first_candidate = pd.to_numeric(
            pd.Series([pair.get("first_feasible_nfe_candidate")]), errors="coerce"
        ).iloc[0]
        first_reference = pd.to_numeric(
            pd.Series([pair.get("first_feasible_nfe_reference")]), errors="coerce"
        ).iloc[0]
        end_candidate = pd.to_numeric(
            pd.Series([pair.get("search_nfe_candidate")]), errors="coerce"
        ).iloc[0]
        end_reference = pd.to_numeric(
            pd.Series([pair.get("search_nfe_reference")]), errors="coerce"
        ).iloc[0]
        if not all(
            np.isfinite(value)
            for value in (first_candidate, first_reference, end_candidate, end_reference)
        ):
            continue
        start_nfe = int(max(first_candidate, first_reference))
        end_nfe = int(min(end_candidate, end_reference))
        try:
            candidate_history = pd.read_csv(
                Path(pair["path_candidate"]) / "history.csv"
            )
            reference_history = pd.read_csv(
                Path(pair["path_reference"]) / "history.csv"
            )
            candidate_value = step_auc(candidate_history, start_nfe, end_nfe)
            reference_value = step_auc(reference_history, start_nfe, end_nfe)
        except (FileNotFoundError, KeyError, ValueError, pd.errors.ParserError):
            continue
        if np.isfinite(candidate_value) and np.isfinite(reference_value):
            candidate_auc.append(candidate_value)
            reference_auc.append(reference_value)
    return _prefix_statistics(
        paired_cost_statistics(candidate_auc, reference_auc), "auc"
    )


def collect_runs(experiment_id, results_root=DEFAULT_RESULTS_ROOT):
    root = Path(results_root) / experiment_id
    rows = []
    if not root.exists():
        return pd.DataFrame()
    seen = set()
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for task in manifest.get("tasks", []):
            config = task.get("config", {})
            method_id = task.get("method_id", config.get("method_id"))
            seed = int(task.get("seed", config.get("seed")))
            relative_path = task.get(
                "relative_path", "%s/seed_%d" % (method_id, seed)
            )
            directory = root / relative_path
            rows.append(
                _collect_run_row(
                    experiment_id,
                    config,
                    directory,
                    manifest_status=task.get("status", "pending"),
                )
            )
            seen.add((method_id, seed))
    for config_path in sorted(root.glob("*/seed_*/config.json")):
        config = read_json(config_path)
        key = (config["method_id"], int(config["seed"]))
        if key not in seen:
            rows.append(_collect_run_row(experiment_id, config, config_path.parent))
    frame = pd.DataFrame(rows)
    defaults = {
        "resolved": False,
        "started": False,
        "unstarted": True,
        "infrastructure_missing": True,
        "feasible": False,
        "best_cost": float("nan"),
        "first_feasible_nfe": float("nan"),
        "search_nfe": float("nan"),
        "wall_seconds": float("nan"),
        "selector_seconds": float("nan"),
    }
    for name, default in defaults.items():
        if name not in frame:
            frame[name] = default
        else:
            frame[name] = frame[name].fillna(default)
    return frame


def _collect_run_row(
    experiment_id, config, directory, manifest_status=None
):
    config_path = Path(directory) / "config.json"
    started = config_path.exists() or manifest_status not in (None, "pending")
    row = {
        "experiment_id": experiment_id,
        "group": config.get("group"),
        "method": config.get("method"),
        "method_id": config.get("method_id", config.get("method")),
        "variant": config.get("variant", "default"),
        "seed": int(config["seed"]),
        "path": str(directory),
        "manifest_status": manifest_status,
        "started": started,
        "unstarted": not started,
        "resolved": False,
        "infrastructure_missing": True,
    }
    summary_path = Path(directory) / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary.get("status") == "complete":
            row.update(summary)
            row["resolved"] = True
            row["started"] = True
            row["unstarted"] = False
            row["infrastructure_missing"] = False
    return row


def summarize_methods(runs, expected_seeds=None):
    rows = []
    if runs.empty:
        return pd.DataFrame()
    for method_id, group in runs.groupby("method_id", sort=False):
        resolved = group[group["resolved"]]
        feasible_mask = resolved["feasible"].fillna(False).astype(bool)
        feasible = int(feasible_mask.sum())
        total = len(resolved)
        low, high = wilson_interval(feasible, total)
        feasible_costs = pd.to_numeric(
            resolved.loc[feasible_mask, "best_cost"], errors="coerce"
        ).dropna()
        planned = len(group) if expected_seeds is None else expected_seeds
        rows.append(
            {
                "method_id": method_id,
                "planned": planned,
                "started": int(group["started"].sum()),
                "unstarted": int(group["unstarted"].sum()),
                "resolved": total,
                "infrastructure_missing": max(planned - total, 0),
                "feasible": feasible,
                "feasible_rate": feasible / total if total else float("nan"),
                "wilson_low": low,
                "wilson_high": high,
                "median_feasible_cost": float(feasible_costs.median()) if len(feasible_costs) else float("nan"),
                "iqr_feasible_cost": float(feasible_costs.quantile(0.75) - feasible_costs.quantile(0.25)) if len(feasible_costs) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def paired_comparisons(runs, comparisons=COMPARISONS):
    rows = []
    feasibility_p = {}
    cost_p = {}
    for candidate, reference in comparisons:
        name = "%s vs %s" % (candidate, reference)
        candidate_rows = runs[(runs["method_id"] == candidate) & runs["resolved"]]
        reference_rows = runs[(runs["method_id"] == reference) & runs["resolved"]]
        paired = candidate_rows.merge(reference_rows, on="seed", suffixes=("_candidate", "_reference"))
        if paired.empty:
            mcnemar = {
                "candidate_only_feasible": 0,
                "reference_only_feasible": 0,
                "discordant": 0,
                "p_value": None,
            }
            cost_stats = paired_cost_statistics([], [])
            auc_stats = _paired_auc_statistics(paired)
        else:
            mcnemar = exact_mcnemar(
                paired["feasible_candidate"].fillna(False).astype(bool),
                paired["feasible_reference"].fillna(False).astype(bool),
            )
            jointly_feasible = paired[
                paired["feasible_candidate"].fillna(False).astype(bool)
                & paired["feasible_reference"].fillna(False).astype(bool)
            ]
            cost_stats = paired_cost_statistics(
                jointly_feasible["best_cost_candidate"],
                jointly_feasible["best_cost_reference"],
            )
            auc_stats = _paired_auc_statistics(jointly_feasible)
        row = {
            "comparison": name,
            "candidate": candidate,
            "reference": reference,
            "resolved_pairs": len(paired),
            "mcnemar_candidate_only": mcnemar["candidate_only_feasible"],
            "mcnemar_reference_only": mcnemar["reference_only_feasible"],
            "mcnemar_p": mcnemar["p_value"],
        }
        row.update(cost_stats)
        row.update(auc_stats)
        rows.append(row)
        feasibility_p[name] = mcnemar["p_value"]
        cost_p[name] = cost_stats["wilcoxon_p"]

    feasibility_adjusted = holm_adjust(feasibility_p, family_size=len(COMPARISONS))
    cost_adjusted = holm_adjust(cost_p, family_size=len(COMPARISONS))
    for row in rows:
        row["mcnemar_holm_p"] = feasibility_adjusted[row["comparison"]]
        row["wilcoxon_holm_p"] = cost_adjusted[row["comparison"]]
    return pd.DataFrame(rows)


def _jsonl_records(path):
    if not Path(path).exists():
        return []
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def convergence_summary(runs):
    rows = []
    for _, run in runs[runs["resolved"]].iterrows():
        history_path = Path(run["path"]) / "history.csv"
        if not history_path.exists():
            continue
        try:
            history = pd.read_csv(history_path)
        except (ValueError, pd.errors.ParserError):
            continue
        required = {"search_nfe", "best_cost", "feasible"}
        if not required.issubset(history.columns):
            continue
        feasible = history[history["feasible"].astype(str).str.lower() == "true"]
        for _, point in feasible.iterrows():
            rows.append(
                {
                    "method_id": run["method_id"],
                    "seed": run["seed"],
                    "search_nfe": int(point["search_nfe"]),
                    "best_cost": float(point["best_cost"]),
                }
            )
    columns = (
        "method_id", "search_nfe", "runs", "median_best_cost", "q25_best_cost",
        "q75_best_cost",
    )
    if not rows:
        return pd.DataFrame(columns=columns)
    points = pd.DataFrame(rows)
    grouped = points.groupby(["method_id", "search_nfe"], sort=False)["best_cost"]
    summary = grouped.agg(runs="count", median_best_cost="median").reset_index()
    summary["q25_best_cost"] = grouped.quantile(0.25).to_numpy()
    summary["q75_best_cost"] = grouped.quantile(0.75).to_numpy()
    return summary.loc[:, columns]


def phase_reward_summary(runs):
    rows = []
    for _, run in runs[runs["resolved"]].iterrows():
        for action in _jsonl_records(Path(run["path"]) / "actions.jsonl"):
            try:
                reward = float(action.get("relative_improvement"))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(reward):
                continue
            rows.append(
                {
                    "method_id": run["method_id"],
                    "phase": action.get("phase_at_start", "unknown"),
                    "action": action.get("applied_action", "unknown"),
                    "reward": reward,
                }
            )
    columns = (
        "method_id", "phase", "action", "windows", "mean_reward", "median_reward",
        "q25_reward", "q75_reward",
    )
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["method_id", "phase", "action"], sort=False)["reward"]
    summary = grouped.agg(
        windows="count", mean_reward="mean", median_reward="median"
    ).reset_index()
    summary["q25_reward"] = grouped.quantile(0.25).to_numpy()
    summary["q75_reward"] = grouped.quantile(0.75).to_numpy()
    return summary.loc[:, columns]


def overhead_summary(runs):
    columns = (
        "method_id", "runs", "median_wall_seconds", "median_search_seconds",
        "median_selector_seconds", "median_selector_fraction",
    )
    resolved = runs[runs["resolved"]].copy()
    if resolved.empty:
        return pd.DataFrame(columns=columns)
    resolved["wall_seconds"] = pd.to_numeric(resolved["wall_seconds"], errors="coerce")
    resolved["selector_seconds"] = pd.to_numeric(
        resolved["selector_seconds"], errors="coerce"
    )
    resolved["search_seconds"] = (
        resolved["wall_seconds"] - resolved["selector_seconds"]
    ).clip(lower=0.0)
    resolved["selector_fraction"] = np.divide(
        resolved["selector_seconds"],
        resolved["wall_seconds"],
        out=np.full(len(resolved), np.nan),
        where=resolved["wall_seconds"].to_numpy() > 0.0,
    )
    rows = []
    for method_id, group in resolved.groupby("method_id", sort=False):
        rows.append(
            {
                "method_id": method_id,
                "runs": len(group),
                "median_wall_seconds": group["wall_seconds"].median(),
                "median_search_seconds": group["search_seconds"].median(),
                "median_selector_seconds": group["selector_seconds"].median(),
                "median_selector_fraction": group["selector_fraction"].median(),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _usage_value(usage, names):
    if not isinstance(usage, Mapping):
        return 0.0
    for name in names:
        value = usage.get(name)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    return 0.0


def llm_summary(runs):
    columns = (
        "method_id", "runs", "calls", "fallbacks", "invalid_outputs", "timeouts",
        "input_tokens", "output_tokens", "total_tokens", "provider_cost",
        "elapsed_seconds",
    )
    method_rows = []
    llm_runs = runs[runs["resolved"] & runs["method_id"].astype(str).str.startswith("LLM-")]
    for _, run in llm_runs.iterrows():
        calls = _jsonl_records(Path(run["path"]) / "llm_calls.jsonl")
        row = {
            "method_id": run["method_id"],
            "runs": 1,
            "calls": len(calls),
            "fallbacks": 0,
            "invalid_outputs": 0,
            "timeouts": 0,
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "total_tokens": 0.0,
            "provider_cost": 0.0,
            "elapsed_seconds": 0.0,
        }
        for call in calls:
            error_kind = str(call.get("error_kind") or "").lower()
            usage = call.get("usage")
            row["fallbacks"] += int(bool(call.get("fallback_used")))
            row["invalid_outputs"] += int("invalid" in error_kind or "schema" in error_kind)
            row["timeouts"] += int("timeout" in error_kind)
            row["input_tokens"] += _usage_value(
                usage, ("input_tokens", "prompt_tokens")
            )
            row["output_tokens"] += _usage_value(
                usage, ("output_tokens", "completion_tokens")
            )
            total = _usage_value(usage, ("total_tokens",))
            row["total_tokens"] += total or (
                _usage_value(usage, ("input_tokens", "prompt_tokens"))
                + _usage_value(usage, ("output_tokens", "completion_tokens"))
            )
            cost = call.get("provider_cost")
            elapsed = call.get("elapsed_seconds")
            row["provider_cost"] += float(cost) if isinstance(cost, (int, float)) else 0.0
            row["elapsed_seconds"] += float(elapsed) if isinstance(elapsed, (int, float)) else 0.0
        method_rows.append(row)
    if not method_rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(method_rows)
        .groupby("method_id", sort=False, as_index=False)
        .sum(numeric_only=True)
        .loc[:, columns]
    )


def _save_figure(path, draw):
    def writer(temporary):
        figure = draw()
        figure.savefig(temporary, dpi=180, bbox_inches="tight", format="png")
        plt.close(figure)

    atomic_write_via(path, writer)


def render_figures(
    runs, method_summary, pairwise, convergence, phase_rewards, overhead, llm, output_dir
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if method_summary.empty:
        return

    def feasible_plot():
        frame = method_summary.reset_index(drop=True)
        figure, axis = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(frame))
        values = frame["feasible_rate"].fillna(0.0).to_numpy(float)
        low = frame["wilson_low"].fillna(0.0).to_numpy(float)
        high = frame["wilson_high"].fillna(0.0).to_numpy(float)
        lower = np.maximum(values - low, 0.0)
        upper = np.maximum(high - values, 0.0)
        axis.bar(x, values, color="#3B6FB6")
        axis.errorbar(x, values, yerr=[lower, upper], fmt="none", color="black", capsize=3)
        axis.set_xticks(x)
        axis.set_xticklabels(frame["method_id"], rotation=35, ha="right")
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Feasible run rate")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        return figure

    _save_figure(output_dir / "feasible_rate.png", feasible_plot)

    feasible_runs = runs[runs["resolved"] & runs["feasible"].fillna(False).astype(bool)]
    if not feasible_runs.empty:
        def cost_plot():
            methods = list(dict.fromkeys(feasible_runs["method_id"]))
            values = [
                pd.to_numeric(
                    feasible_runs.loc[feasible_runs["method_id"] == method, "best_cost"],
                    errors="coerce",
                ).dropna().to_numpy()
                for method in methods
            ]
            figure, axis = plt.subplots(figsize=(9, 4.5))
            axis.boxplot(values, labels=methods, showfliers=False)
            axis.set_ylabel("Final feasible cost")
            axis.tick_params(axis="x", rotation=35)
            axis.grid(axis="y", alpha=0.25)
            figure.tight_layout()
            return figure

        _save_figure(output_dir / "final_cost.png", cost_plot)

    available = pd.DataFrame()
    if not pairwise.empty:
        available = pairwise[
            (pairwise["joint_feasible_pairs"] > 0)
            & np.isfinite(pairwise["difference_ci_low"])
            & np.isfinite(pairwise["difference_ci_high"])
        ]
    if not available.empty:
        def effect_plot():
            frame = available.reset_index(drop=True)
            y = np.arange(len(frame))
            values = frame["median_difference"].to_numpy(float)
            low = frame["difference_ci_low"].to_numpy(float)
            high = frame["difference_ci_high"].to_numpy(float)
            figure, axis = plt.subplots(figsize=(9, max(4.5, len(frame) * 0.45)))
            axis.errorbar(values, y, xerr=[values - low, high - values], fmt="o", color="#B2473E", capsize=3)
            axis.axvline(0.0, color="black", linewidth=0.8)
            axis.set_yticks(y)
            axis.set_yticklabels(frame["comparison"])
            axis.set_xlabel("Median paired cost difference (candidate - reference)")
            axis.grid(axis="x", alpha=0.25)
            figure.tight_layout()
            return figure

        _save_figure(output_dir / "paired_cost_effects.png", effect_plot)

    action_rows = []
    for path_text in runs.loc[runs["resolved"], "path"]:
        action_path = Path(path_text) / "actions.jsonl"
        if not action_path.exists():
            continue
        method = read_json(Path(path_text) / "summary.json")["method_id"]
        with action_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    action_rows.append({"method_id": method, "action": record["applied_action"]})
    if action_rows:
        action_frame = pd.DataFrame(action_rows)
        table = pd.crosstab(action_frame["method_id"], action_frame["action"], normalize="index")

        def action_plot():
            figure, axis = plt.subplots(figsize=(7, max(3.5, len(table) * 0.4)))
            image = axis.imshow(table.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
            axis.set_xticks(np.arange(len(table.columns)))
            axis.set_xticklabels(table.columns)
            axis.set_yticks(np.arange(len(table.index)))
            axis.set_yticklabels(table.index)
            axis.set_xlabel("Action")
            axis.set_ylabel("Method")
            figure.colorbar(image, ax=axis, label="Window fraction")
            figure.tight_layout()
            return figure

        _save_figure(output_dir / "action_usage.png", action_plot)

    if not convergence.empty:
        def convergence_plot():
            figure, axis = plt.subplots(figsize=(9, 4.8))
            for method_id, frame in convergence.groupby("method_id", sort=False):
                frame = frame.sort_values("search_nfe")
                axis.plot(
                    frame["search_nfe"], frame["median_best_cost"], label=method_id
                )
            axis.set_xlabel("Search NFE")
            axis.set_ylabel("Median best feasible cost")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8, ncol=2)
            figure.tight_layout()
            return figure

        _save_figure(output_dir / "convergence_nfe.png", convergence_plot)

    if not phase_rewards.empty:
        reward_table = phase_rewards.pivot_table(
            index=["method_id", "phase"],
            columns="action",
            values="median_reward",
            aggfunc="first",
        )

        def reward_plot():
            figure, axis = plt.subplots(
                figsize=(7, max(3.5, len(reward_table) * 0.42))
            )
            image = axis.imshow(reward_table.to_numpy(), aspect="auto", cmap="RdYlGn")
            axis.set_xticks(np.arange(len(reward_table.columns)))
            axis.set_xticklabels(reward_table.columns)
            axis.set_yticks(np.arange(len(reward_table.index)))
            axis.set_yticklabels(
                ["%s | %s" % index for index in reward_table.index]
            )
            axis.set_xlabel("Applied action")
            figure.colorbar(image, ax=axis, label="Median phase reward")
            figure.tight_layout()
            return figure

        _save_figure(output_dir / "phase_rewards.png", reward_plot)

    if not overhead.empty:
        def overhead_plot():
            frame = overhead.reset_index(drop=True)
            x = np.arange(len(frame))
            figure, axis = plt.subplots(figsize=(9, 4.5))
            axis.bar(
                x,
                frame["median_search_seconds"],
                label="Search excluding selector",
                color="#4C78A8",
            )
            axis.bar(
                x,
                frame["median_selector_seconds"],
                bottom=frame["median_search_seconds"],
                label="Selector",
                color="#F58518",
            )
            axis.set_xticks(x)
            axis.set_xticklabels(frame["method_id"], rotation=35, ha="right")
            axis.set_ylabel("Median wall time (seconds)")
            axis.legend()
            axis.grid(axis="y", alpha=0.25)
            figure.tight_layout()
            return figure

        _save_figure(output_dir / "selector_overhead.png", overhead_plot)

    if not llm.empty:
        def llm_plot():
            frame = llm.reset_index(drop=True)
            x = np.arange(len(frame))
            figure, axes = plt.subplots(2, 2, figsize=(9, 6.5))
            metrics = (
                ("calls", "Calls"),
                ("fallbacks", "Fallbacks"),
                ("total_tokens", "Tokens"),
                ("provider_cost", "Provider cost"),
            )
            for axis, (column, label) in zip(axes.flat, metrics):
                axis.bar(x, frame[column], color="#4C78A8")
                axis.set_xticks(x)
                axis.set_xticklabels(frame["method_id"], rotation=25, ha="right")
                axis.set_ylabel(label)
                axis.grid(axis="y", alpha=0.25)
            figure.tight_layout()
            return figure

        _save_figure(output_dir / "llm_metrics.png", llm_plot)


def write_report(experiment_id, runs, method_summary, pairwise, path):
    resolved = int(runs["resolved"].sum()) if not runs.empty else 0
    planned = len(runs)
    started = int(runs["started"].sum()) if not runs.empty else 0
    unstarted = int(runs["unstarted"].sum()) if not runs.empty else 0
    lines = [
        "# Case 2 Experiment Summary",
        "",
        "- Experiment: `%s`" % experiment_id,
        "- Resolved runs: `%d/%d`" % (resolved, planned),
        "- Started runs: `%d/%d`" % (started, planned),
        "- Unstarted runs: `%d`" % unstarted,
        "- Infrastructure-missing runs: `%d`" % (planned - resolved),
        "",
        "Positive paired cost differences and positive rank-biserial values mean the candidate is more expensive than the reference.",
        "",
        "Method, paired final-cost/AUC, NFE convergence, phase reward, overhead, and LLM tables are stored beside this report.",
        "Bootstrap intervals and Wilcoxon cost inference are omitted when fewer than 10 jointly feasible pairs are available.",
        "",
    ]
    atomic_write_text(path, "\n".join(lines))


def analyze_experiment(experiment_id, results_root=DEFAULT_RESULTS_ROOT, diagnostic=False):
    results_root = _require_case2_path(results_root)
    runs = collect_runs(experiment_id, results_root)
    if runs.empty:
        raise FileNotFoundError("No run configs found for experiment %s" % experiment_id)
    method_summary = summarize_methods(runs)
    pairwise = pd.DataFrame() if diagnostic else paired_comparisons(runs)
    convergence = convergence_summary(runs)
    phase_rewards = phase_reward_summary(runs)
    overhead = overhead_summary(runs)
    llm = llm_summary(runs)
    aggregate = _require_case2_path(Path(results_root) / experiment_id / "aggregate")
    aggregate.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(aggregate / "runs.csv", runs.columns, runs.to_dict("records"))
    atomic_write_csv(
        aggregate / ("diagnostic_summary.csv" if diagnostic else "method_summary.csv"),
        method_summary.columns,
        method_summary.to_dict("records"),
    )
    if not diagnostic:
        atomic_write_csv(
            aggregate / "pairwise_stats.csv", pairwise.columns, pairwise.to_dict("records")
        )
    for filename, frame in (
        ("convergence_nfe.csv", convergence),
        ("phase_rewards.csv", phase_rewards),
        ("selector_overhead.csv", overhead),
        ("llm_metrics.csv", llm),
    ):
        atomic_write_csv(aggregate / filename, frame.columns, frame.to_dict("records"))
    render_figures(
        runs,
        method_summary,
        pairwise,
        convergence,
        phase_rewards,
        overhead,
        llm,
        aggregate / "figures",
    )
    write_report(experiment_id, runs, method_summary, pairwise, aggregate / "report.md")
    return runs, method_summary, pairwise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate Case 2 experiment results")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args(argv)
    analyze_experiment(
        args.experiment_id, Path(args.results_root), diagnostic=args.diagnostic
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
