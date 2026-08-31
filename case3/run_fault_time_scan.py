from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    from case3 import solve_case3 as model
except ModuleNotFoundError:
    import solve_case3 as model


FAULT_START_HOURS = (5, 8, 11, 14, 17, 20)
FAULT_DURATION = 4
ESS_CAPACITY_MWH = 10.0
G2_MAX_MW = 15.0
G1_FAULT_MAX_MW = 0.0
SEED = 20260826


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def solve_normal_plans(p_vital, p_nonvital, p_pv):
    plans = {}
    for strategy in ("no_reserve", "dynamic_reserve"):
        _, result, _ = model.solve_ra_lshade(
            p_vital,
            p_nonvital,
            p_pv,
            strategy,
            SEED,
        )
        plans[strategy] = result
    return plans


def minimum_shortfall_diagnostic(
    normal_result, p_vital, p_nonvital, p_pv, start_hour
):
    hours = np.arange(start_hour, start_hour + FAULT_DURATION)
    energy = float(normal_result["energy_ess"][start_hour - 1])
    p_g2 = min(
        model.G2_MAX,
        float(normal_result["p_g2"][start_hour - 1]) + model.G2_RAMP,
    )
    nonvital_shed = []
    vital_shed = []
    ess_output = []
    g2_output = []
    remaining_energy = []
    for index, hour in enumerate(hours):
        if index > 0:
            p_g2 = min(model.G2_MAX, p_g2 + model.G2_RAMP)
        demand = (
            p_vital[hour]
            + p_nonvital[hour]
            + normal_result["p_propulsion"][hour]
            - p_pv[hour]
        )
        maximum_energy_output = max(
            0.0, (energy - model.ESS_ENERGY_MIN) * model.ESS_EFFICIENCY_OUT
        )
        remaining_demand = max(
            0.0, demand - model.FAULT_G1_MAX - p_g2
        )
        p_ess = min(
            model.ESS_POWER_MAX, maximum_energy_output, remaining_demand
        )
        total_shortfall = max(
            0.0, demand - model.FAULT_G1_MAX - p_g2 - p_ess
        )
        shed_nonvital = min(float(p_nonvital[hour]), total_shortfall)
        shed_vital = max(0.0, total_shortfall - shed_nonvital)
        energy -= p_ess / model.ESS_EFFICIENCY_OUT * model.DELTA_T
        nonvital_shed.append(shed_nonvital)
        vital_shed.append(shed_vital)
        ess_output.append(p_ess)
        g2_output.append(p_g2)
        remaining_energy.append(energy)
    return {
        "hours": hours,
        "p_g2": np.array(g2_output),
        "p_ess": np.array(ess_output),
        "energy_ess": np.array(remaining_energy),
        "p_shed": np.array(nonvital_shed),
        "p_vital_shortfall": np.array(vital_shed),
        "pre_fault_energy": float(normal_result["energy_ess"][start_hour - 1]),
        "shed_energy": float(np.sum(nonvital_shed) * model.DELTA_T),
        "vital_shortfall_energy": float(np.sum(vital_shed) * model.DELTA_T),
    }


def run_scan(base_dir):
    model.configure_ess_capacity(ESS_CAPACITY_MWH)
    model.configure_g2_max(G2_MAX_MW)
    model.configure_fault_g1_max(G1_FAULT_MAX_MW)
    _, p_vital, p_nonvital, p_pv = model.load_input_data(base_dir.parent / "data.md")
    plans = solve_normal_plans(p_vital, p_nonvital, p_pv)

    result_rows = []
    hourly_rows = []
    for strategy, normal_result in plans.items():
        normal_validation = model.normal_validation(normal_result, strategy)
        for start_hour in FAULT_START_HOURS:
            status = "OPTIMAL"
            try:
                fault = model.solve_fault(
                    normal_result,
                    p_vital,
                    p_nonvital,
                    p_pv,
                    fault_start_hour=start_hour,
                    fault_duration=FAULT_DURATION,
                )
                fault["p_vital_shortfall"] = np.zeros(FAULT_DURATION)
                fault["vital_shortfall_energy"] = 0.0
                fault_balance_error = float(
                    np.max(np.abs(fault["balance_residual"]))
                )
                minimum_slack = fault["minimum_inequality_slack"]
            except RuntimeError:
                fault = minimum_shortfall_diagnostic(
                    normal_result, p_vital, p_nonvital, p_pv, start_hour
                )
                if fault["vital_shortfall_energy"] <= 1.0e-8:
                    raise
                status = "INFEASIBLE_CRITICAL_LOAD"
                fault_balance_error = np.nan
                minimum_slack = np.nan
            nonvital_energy = float(
                np.sum(p_nonvital[fault["hours"]]) * model.DELTA_T
            )
            loss_rate = fault["shed_energy"] / nonvital_energy
            vital_energy = float(
                np.sum(p_vital[fault["hours"]]) * model.DELTA_T
            )
            vital_loss_rate = fault["vital_shortfall_energy"] / vital_energy
            result_rows.append(
                {
                    "strategy": strategy,
                    "status": status,
                    "fault_start_hour": start_hour,
                    "fault_end_hour": start_hour + FAULT_DURATION - 1,
                    "fault_duration_h": FAULT_DURATION,
                    "normal_cost": normal_result["total_cost"],
                    "fault_pre_soc": fault["pre_fault_energy"] / model.ESS_ENERGY_MAX,
                    "fault_end_soc": fault["energy_ess"][-1] / model.ESS_ENERGY_MAX,
                    "fault_ess_output_mwh": float(
                        np.sum(np.maximum(0.0, fault["p_ess"])) * model.DELTA_T
                    ),
                    "fault_g2_max_mw": float(np.max(fault["p_g2"])),
                    "nonvital_energy_mwh": nonvital_energy,
                    "shed_energy_mwh": fault["shed_energy"],
                    "loss_rate": loss_rate,
                    "vital_shortfall_energy_mwh": fault[
                        "vital_shortfall_energy"
                    ],
                    "vital_loss_rate": vital_loss_rate,
                    "load_retention": 1.0 - loss_rate,
                    "normal_balance_error_mw": normal_validation["balance_error"],
                    "fault_balance_error_mw": fault_balance_error,
                    "minimum_inequality_slack": minimum_slack,
                }
            )
            for index, hour in enumerate(fault["hours"]):
                hourly_rows.append(
                    {
                        "strategy": strategy,
                        "fault_start_hour": start_hour,
                        "hour": int(hour),
                        "p_vital_mw": p_vital[hour],
                        "p_nonvital_mw": p_nonvital[hour],
                        "p_pv_mw": p_pv[hour],
                        "v_kn": normal_result["speeds"][hour],
                        "p_propulsion_mw": normal_result["p_propulsion"][hour],
                        "p_g1_mw": 0.0
                        if status != "OPTIMAL"
                        else fault["p_g1"][index],
                        "p_g2_mw": fault["p_g2"][index],
                        "p_ess_mw": fault["p_ess"][index],
                        "energy_ess_mwh": fault["energy_ess"][index],
                        "p_shed_mw": fault["p_shed"][index],
                        "p_vital_shortfall_mw": fault["p_vital_shortfall"][index],
                        "balance_residual_mw": ""
                        if status != "OPTIMAL"
                        else fault["balance_residual"][index],
                    }
                )

    output_dir = base_dir / "fault_time_scan"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "fault_time_results.csv", result_rows)
    write_csv(output_dir / "fault_time_hourly.csv", hourly_rows)
    write_summary(output_dir / "summary.md", result_rows)
    return result_rows


def write_summary(path, rows):
    lookup = {
        (row["strategy"], row["fault_start_hour"]): row for row in rows
    }
    lines = [
        "# 最严重设备参数下的故障时刻扫描",
        "",
        "| 参数 | 取值 |",
        "|---|---:|",
        f"| ESS 容量 / 最大功率 | {ESS_CAPACITY_MWH:g} MWh / {model.ESS_POWER_MAX:g} MW |",
        f"| G2 最大出力 / 爬坡 | {G2_MAX_MW:g} MW / {model.G2_RAMP:g} MW/h |",
        f"| G1 故障后最大出力 | {G1_FAULT_MAX_MW:g} MW |",
        f"| 故障持续时间 | {FAULT_DURATION} h |",
        f"| 故障起点 | {', '.join(str(value) for value in FAULT_START_HOURS)} h |",
        "",
        "| 故障区间 | No reserve 状态 | No reserve 非重要失负荷 | No reserve 重要负荷缺额 | Dynamic 非重要失负荷 | Dynamic 重要负荷缺额 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for start_hour in FAULT_START_HOURS:
        no = lookup[("no_reserve", start_hour)]
        dynamic = lookup[("dynamic_reserve", start_hour)]
        lines.append(
            f"| {start_hour}–{start_hour + FAULT_DURATION - 1} h | "
            f"{no['status']} | "
            f"{no['shed_energy_mwh']:.6f} MWh ({100 * no['loss_rate']:.3f}%) | "
            f"{no['vital_shortfall_energy_mwh']:.6f} MWh | "
            f"{dynamic['shed_energy_mwh']:.6f} MWh ({100 * dynamic['loss_rate']:.3f}%) | "
            f"{dynamic['vital_shortfall_energy_mwh']:.6f} MWh |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    base_dir = Path(__file__).resolve().parent
    rows = run_scan(base_dir)
    worst = max(rows, key=lambda row: row["loss_rate"])
    print(
        f"worst: {worst['strategy']} at hour={worst['fault_start_hour']}, "
        f"shed={worst['shed_energy_mwh']:.6f} MWh, "
        f"loss_rate={100 * worst['loss_rate']:.3f}%"
    )


if __name__ == "__main__":
    main()
