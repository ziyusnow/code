from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    from case3 import solve_case3 as model
except ModuleNotFoundError:
    import solve_case3 as model


ESS_CAPACITIES = (10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
G2_CAPACITIES = (15.0, 16.0, 17.0, 18.0, 19.0, 20.0)
G1_FAULT_LIMITS = (6.0, 2.0, 0.0)
BATCH_SEED = 20260826
RETRY_SEEDS = (20260826, 20260827, 20260828, 20260829, 20260830)


def _strategy_row(
    ess_capacity,
    g2_capacity,
    g1_fault_limit,
    strategy,
    p1_seed,
    p2_seed,
    result,
    fault,
):
    return {
        "ess_capacity_mwh": ess_capacity,
        "ess_power_max_mw": model.ESS_POWER_MAX,
        "g2_max_mw": g2_capacity,
        "g2_ramp_mw_per_h": model.G2_RAMP,
        "g1_fault_max_mw": g1_fault_limit,
        "strategy": strategy,
        "p1_seed": p1_seed,
        "p2_seed": p2_seed,
        "normal_cost": float(result["total_cost"]),
        "normal_cv_base": float(result["cv_base"]),
        "normal_cv_res": float(result["cv_res"]),
        "fault_cv": float(fault["cv_f"]),
        "reserve_energy_max_mwh": float(np.nanmax(result["reserve_energy"])),
        "reserve_power_max_mw": float(np.nanmax(result["reserve_power_max"])),
        "fault_pre_soc": float(fault["pre_fault_energy"] / model.ESS_ENERGY_MAX),
        "fault_end_soc": float(fault["energy_ess"][-1] / model.ESS_ENERGY_MAX),
        "fault_ess_output_mwh": float(np.sum(np.maximum(0.0, fault["p_ess"]))),
        "fault_g2_max_mw": float(np.max(fault["p_g2"])),
        "shed_energy_mwh": float(fault["shed_energy"]),
        "load_retention": float(fault["load_retention"]),
    }


def _solve_normal_with_retries(
    p_vital, p_nonvital, p_pv, strategy, np_max, np_min, iterations
):
    last_error = None
    for seed in RETRY_SEEDS:
        try:
            position, result, _ = model.solve_ra_lshade(
                p_vital,
                p_nonvital,
                p_pv,
                strategy,
                seed,
                np_max=np_max,
                np_min=np_min,
                iterations=iterations,
            )
            return seed, position, result
        except RuntimeError as error:
            last_error = error
    raise RuntimeError(
        f"P1 RA-LSHADE did not find a feasible {strategy} solution"
    ) from last_error


def _solve_fault_with_retries(
    normal_result, p_vital, p_nonvital, p_pv, np_max, np_min, iterations
):
    last_error = None
    for seed in RETRY_SEEDS:
        try:
            fault = model.solve_fault(
                normal_result,
                p_vital,
                p_nonvital,
                p_pv,
                seed=seed,
                np_max=np_max,
                np_min=np_min,
                iterations=iterations,
            )
            return seed, fault
        except RuntimeError as error:
            last_error = error
    raise RuntimeError("P2 RA-LSHADE did not find a feasible solution") from last_error


def solve_design_point(args):
    ess_capacity, g2_capacity, np_max, np_min, iterations = args
    model.configure_ess_capacity(ess_capacity)
    model.configure_g2_max(g2_capacity)
    root = Path(__file__).resolve().parent.parent
    _, p_vital, p_nonvital, p_pv = model.load_input_data(root / "data.md")

    model.configure_fault_g1_max(G1_FAULT_LIMITS[0])
    no_p1_seed, no_position, _ = _solve_normal_with_retries(
        p_vital,
        p_nonvital,
        p_pv,
        "no_reserve",
        np_max,
        np_min,
        iterations,
    )

    rows = []
    for fault_limit in G1_FAULT_LIMITS:
        model.configure_fault_g1_max(fault_limit)
        no_result_batch = model.evaluate_normal(
            no_position, p_vital, p_nonvital, p_pv, "no_reserve", model.LAMBDA_R_MAX
        )
        no_result = {key: value[0] for key, value in no_result_batch.items()}
        no_p2_seed, no_fault = _solve_fault_with_retries(
            no_result,
            p_vital,
            p_nonvital,
            p_pv,
            np_max,
            np_min,
            iterations,
        )
        rows.append(
            _strategy_row(
                ess_capacity,
                g2_capacity,
                fault_limit,
                "no_reserve",
                no_p1_seed,
                no_p2_seed,
                no_result,
                no_fault,
            )
        )

        dynamic_p1_seed, _, dynamic_result = _solve_normal_with_retries(
            p_vital,
            p_nonvital,
            p_pv,
            "dynamic_reserve",
            np_max,
            np_min,
            iterations,
        )
        dynamic_p2_seed, dynamic_fault = _solve_fault_with_retries(
            dynamic_result,
            p_vital,
            p_nonvital,
            p_pv,
            np_max,
            np_min,
            iterations,
        )
        rows.append(
            _strategy_row(
                ess_capacity,
                g2_capacity,
                fault_limit,
                "dynamic_reserve",
                dynamic_p1_seed,
                dynamic_p2_seed,
                dynamic_result,
                dynamic_fault,
            )
        )
    return rows


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows(strategy_rows):
    indexed = {
        (
            row["ess_capacity_mwh"],
            row["g2_max_mw"],
            row["g1_fault_max_mw"],
            row["strategy"],
        ): row
        for row in strategy_rows
    }
    rows = []
    for ess_capacity in ESS_CAPACITIES:
        for g2_capacity in G2_CAPACITIES:
            for fault_limit in G1_FAULT_LIMITS:
                no = indexed[(ess_capacity, g2_capacity, fault_limit, "no_reserve")]
                dynamic = indexed[
                    (ess_capacity, g2_capacity, fault_limit, "dynamic_reserve")
                ]
                rows.append(
                    {
                        "ess_capacity_mwh": ess_capacity,
                        "g2_max_mw": g2_capacity,
                        "g1_fault_max_mw": fault_limit,
                        "no_reserve_cost": no["normal_cost"],
                        "dynamic_reserve_cost": dynamic["normal_cost"],
                        "cost_increase_pct": 100.0
                        * (dynamic["normal_cost"] - no["normal_cost"])
                        / no["normal_cost"],
                        "no_reserve_pre_soc": no["fault_pre_soc"],
                        "dynamic_reserve_pre_soc": dynamic["fault_pre_soc"],
                        "no_reserve_shed_mwh": no["shed_energy_mwh"],
                        "dynamic_reserve_shed_mwh": dynamic["shed_energy_mwh"],
                        "shed_reduction_mwh": no["shed_energy_mwh"]
                        - dynamic["shed_energy_mwh"],
                    }
                )
    return rows


def _format_matrix(comparisons, fault_limit):
    lookup = {
        (row["ess_capacity_mwh"], row["g2_max_mw"]): row
        for row in comparisons
        if row["g1_fault_max_mw"] == fault_limit
    }
    lines = [
        f"## G1 故障后上限 {fault_limit:g} MW",
        "",
        "单元格为 `No reserve 失负荷 -> Dynamic reserve 失负荷`，单位 MWh。",
        "",
        "| ESS容量 / G2上限 | "
        + " | ".join(f"{value:g} MW" for value in G2_CAPACITIES)
        + " |",
        "|---|" + "---:|" * len(G2_CAPACITIES),
    ]
    for ess_capacity in ESS_CAPACITIES:
        cells = []
        for g2_capacity in G2_CAPACITIES:
            row = lookup[(ess_capacity, g2_capacity)]
            cells.append(
                f"{row['no_reserve_shed_mwh']:.3f} -> "
                f"{row['dynamic_reserve_shed_mwh']:.3f}"
            )
        lines.append(f"| {ess_capacity:g} MWh | " + " | ".join(cells) + " |")
    return lines


def write_summary(path, comparisons, strategy_rows, np_max, np_min, iterations):
    improved = [row for row in comparisons if row["shed_reduction_mwh"] > 1.0e-7]
    eliminated = [
        row
        for row in improved
        if row["dynamic_reserve_shed_mwh"] <= 1.0e-7
        and row["no_reserve_shed_mwh"] > 1.0e-7
    ]
    p1_retried = [row for row in strategy_rows if row["p1_seed"] != BATCH_SEED]
    p2_retried = [row for row in strategy_rows if row["p2_seed"] != BATCH_SEED]
    complete_fault_rows = [
        row for row in comparisons if row["g1_fault_max_mw"] == 0.0
    ]
    complete_fault_shed = [row["no_reserve_shed_mwh"] for row in complete_fault_rows]
    complete_fault_dynamic_shed = [
        row["dynamic_reserve_shed_mwh"] for row in complete_fault_rows
    ]
    complete_fault_cost = [row["cost_increase_pct"] for row in complete_fault_rows]
    lines = [
        "# Case 3 设备设计参数批量计算",
        "",
        "| 设计量 | 取值 | 固定条件 |",
        "|---|---|---|",
        f"| ESS | 容量 `10、12、14、16、18、20 MWh` | 最大功率 `{model.ESS_POWER_MAX:g} MW`，初始/最低 SOC `0.5/0.2` |",
        "| G2 | 最大出力 `15、16、17、18、19、20 MW` | 爬坡 `3 MW/h` |",
        "| G1 故障 | 最大可用出力 `6、2、0 MW` | 正常额定上限 `10 MW` |",
        "",
        f"- P1/P2 均采用 RA-LSHADE，预算为 `NP={np_max}->{np_min}, K={iterations}`；首选种子 `{BATCH_SEED}`，失败后按固定种子序列重试。P1/P2 分别有 `{len(p1_retried)}/{len(p2_retried)}` 条结果使用后续种子。",
        f"- Dynamic reserve 降低失负荷的组合：`{len(improved)}/{len(comparisons)}`；完全消除原有失负荷的组合：`{len(eliminated)}/{len(comparisons)}`。",
        f"- G1 完全停机场景中，No reserve 失负荷为 `{min(complete_fault_shed):.3f}–{max(complete_fault_shed):.3f} MWh`，Dynamic reserve 为 `{min(complete_fault_dynamic_shed):.3f}–{max(complete_fault_dynamic_shed):.3f} MWh`，正常成本变化为 `{min(complete_fault_cost):.3f}%–{max(complete_fault_cost):.3f}%`。",
        "- 本表用于设备参数筛选；单种子元启发式结果可能有局部非单调波动，正式统计结论仍需对选定代表性组合进行多种子稳定性实验。",
        "",
    ]
    for fault_limit in G1_FAULT_LIMITS:
        lines.extend(_format_matrix(comparisons, fault_limit))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run the Case 3 design grid")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be at least 1")

    if args.smoke:
        design_points = [(10.0, 15.0)]
        np_max, np_min, iterations = model.NP_MAX, model.NP_MIN, model.MAX_ITERATIONS
        output_dir_name = "design_grid_smoke"
    else:
        design_points = [
            (ess_capacity, g2_capacity)
            for ess_capacity in ESS_CAPACITIES
            for g2_capacity in G2_CAPACITIES
        ]
        np_max, np_min, iterations = model.NP_MAX, model.NP_MIN, model.MAX_ITERATIONS
        output_dir_name = "design_grid"

    jobs = [
        (ess_capacity, g2_capacity, np_max, np_min, iterations)
        for ess_capacity, g2_capacity in design_points
    ]
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / output_dir_name
    progress_dir = output_dir / "progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    strategy_rows = []
    if args.workers == 1:
        for completed, job in enumerate(jobs, start=1):
            ess_capacity, g2_capacity = job[:2]
            progress_path = progress_dir / (
                f"ess{ess_capacity:g}_g2{g2_capacity:g}.csv"
            )
            point_rows = solve_design_point(job)
            _write_csv(progress_path, point_rows)
            strategy_rows.extend(point_rows)
            print(
                f"completed {completed}/{len(jobs)}: "
                f"ESS={ess_capacity:g} MWh, G2={g2_capacity:g} MW",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(solve_design_point, job): job[:2] for job in jobs}
            for completed, future in enumerate(as_completed(futures), start=1):
                ess_capacity, g2_capacity = futures[future]
                point_rows = future.result()
                progress_path = progress_dir / (
                    f"ess{ess_capacity:g}_g2{g2_capacity:g}.csv"
                )
                _write_csv(progress_path, point_rows)
                strategy_rows.extend(point_rows)
                print(
                    f"completed {completed}/{len(jobs)}: "
                    f"ESS={ess_capacity:g} MWh, G2={g2_capacity:g} MW",
                    flush=True,
                )

    strategy_rows.sort(
        key=lambda row: (
            row["ess_capacity_mwh"],
            row["g2_max_mw"],
            -row["g1_fault_max_mw"],
            row["strategy"],
        )
    )
    comparisons = comparison_rows(strategy_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "strategy_results.csv", strategy_rows)
    _write_csv(output_dir / "strategy_comparison.csv", comparisons)
    write_summary(
        output_dir / "summary.md",
        comparisons,
        strategy_rows,
        np_max,
        np_min,
        iterations,
    )
    print(f"wrote {len(comparisons)} comparisons to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
