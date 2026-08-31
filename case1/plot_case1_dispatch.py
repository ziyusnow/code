from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 2.0
plt.rcParams["legend.frameon"] = False


BALANCE_TOLERANCE_MW = 1.0e-6
COLORS = {
    "g1": "#0F4D92",
    "g2": "#42949E",
    "demand": "#272727",
}


def read_schedule(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) != 24:
        raise ValueError(f"Expected 24 hourly rows, found {len(rows)}")

    values = {
        key: np.array([float(row[key]) for row in rows])
        for key in rows[0]
    }
    hours = values["hour"].astype(int)
    if not np.array_equal(hours, np.arange(24)):
        raise ValueError("The hour column must be sorted from 0 through 23")
    if not all(np.isfinite(column).all() for column in values.values()):
        raise ValueError("The schedule contains NaN or infinite values")

    total_demand = values["p_load_mw"] + values["p_propulsion_mw"]
    total_generation = values["p_g1_mw"] + values["p_g2_mw"]
    residual = np.max(np.abs(total_generation - total_demand))
    if residual > BALANCE_TOLERANCE_MW:
        raise ValueError(f"Power balance residual is {residual:.3e} MW")
    if np.any(values["p_g1_mw"] < 0) or np.any(values["p_g2_mw"] < 0):
        raise ValueError("Generator output cannot be negative in CASE1")

    return hours, values, total_demand, residual


def make_figure(hours, values, total_demand):
    fig, ax = plt.subplots(figsize=(9.0, 5.2), facecolor="white")
    ax.set_facecolor("white")

    g1_bars = ax.bar(
        hours,
        values["p_g1_mw"],
        width=0.76,
        color=COLORS["g1"],
        edgecolor="white",
        linewidth=0.5,
        label="Generator 1",
        zorder=2,
    )
    g2_bars = ax.bar(
        hours,
        values["p_g2_mw"],
        bottom=values["p_g1_mw"],
        width=0.76,
        color=COLORS["g2"],
        edgecolor="white",
        linewidth=0.5,
        label="Generator 2",
        zorder=2,
    )
    demand_line, = ax.plot(
        hours,
        total_demand,
        color=COLORS["demand"],
        linewidth=2.4,
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgecolor=COLORS["demand"],
        markeredgewidth=1.2,
        label="Total demand",
        zorder=3,
    )

    ax.set_xlim(-0.65, 23.65)
    ax.set_ylim(0.0, 20.0)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
    ax.set_xticks(np.arange(24), minor=True)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Time (h)", labelpad=9)
    ax.set_ylabel("Power (MW)", labelpad=10)
    ax.tick_params(axis="both", which="major", direction="out", length=6, width=1.6)
    ax.tick_params(axis="x", which="minor", direction="out", length=3, width=1.0)
    ax.spines["left"].set_color("#272727")
    ax.spines["bottom"].set_color("#272727")

    ax.legend(
        handles=[g1_bars, g2_bars, demand_line],
        labels=["Generator 1", "Generator 2", "Total demand"],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.015),
        ncol=3,
        fontsize=12,
        handlelength=1.7,
        columnspacing=1.5,
        borderaxespad=0.0,
    )

    fig.tight_layout(pad=1.2)
    return fig


def main():
    base_dir = Path(__file__).resolve().parent
    hours, values, total_demand, residual = read_schedule(
        base_dir / "case1_schedule.csv"
    )
    figure = make_figure(hours, values, total_demand)

    output_dir = base_dir / "figures"
    output_dir.mkdir(exist_ok=True)
    output_base = output_dir / "case1_device_dispatch"
    figure.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"max_power_balance_residual_mw={residual:.3e}")
    print(f"saved={output_base}.svg/.pdf/.png")


if __name__ == "__main__":
    main()
