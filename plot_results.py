"""
plot_results.py
----------------
Reads results.csv and plots how the crash-rate gap between Architecture A
and B changes with the "power" of g (network size, training budget).

Usage:
    python plot_results.py [path_to_csv]
    (defaults to config.RESULTS_CSV_PATH; pass results/results_quick_test.csv
    to smoke-test this file before the real sweep has produced any data)

Requires: pip install pandas matplotlib
"""

import sys
import ast

import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_CSV_PATH, OVERRIDE_WEIGHTS


def load(path):
    df = pd.read_csv(path)
    # network_size is stored as the string "[64]"; parse it back to a list,
    # and keep a scalar copy (its one hidden-layer width) for plotting.
    df["network_size"] = df["network_size"].apply(ast.literal_eval)
    df["network_size_scalar"] = df["network_size"].apply(lambda x: x[0])
    return df


def compute_gap(df, condition):
    """
    gap = crash_rate_A - crash_rate_B for a given condition, per
    (network_size, training_timesteps) cell, averaged (mean + std) over
    seeds. A positive gap means A crashes more than B.
    """
    sub = df[df["condition"] == condition]
    a = sub[sub["architecture"] == "A"].set_index(
        ["network_size_scalar", "training_timesteps", "seed"])["crash_rate"]
    b = sub[sub["architecture"] == "B"].set_index(
        ["network_size_scalar", "training_timesteps", "seed"])["crash_rate"]
    gap = (a - b).rename("gap").reset_index()
    return (gap.groupby(["network_size_scalar", "training_timesteps"])["gap"]
            .agg(["mean", "std"]).reset_index())


def plot_gap_vs_network_size(gap_df, condition, ax):
    for budget, sub in gap_df.groupby("training_timesteps"):
        sub = sub.sort_values("network_size_scalar")
        ax.errorbar(sub["network_size_scalar"], sub["mean"], yerr=sub["std"],
                    marker="o", label=f"budget={budget}")
    ax.set_xscale("log")
    ax.set_xlabel("network size (hidden units)")
    ax.set_ylabel("crash-rate gap (A - B)")
    ax.set_title(f"Gap vs network size ({condition})")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend(fontsize="small")


def plot_gap_vs_budget(gap_df, condition, ax):
    for net_size, sub in gap_df.groupby("network_size_scalar"):
        sub = sub.sort_values("training_timesteps")
        ax.errorbar(sub["training_timesteps"], sub["mean"], yerr=sub["std"],
                    marker="o", label=f"net_size={net_size}")
    ax.set_xscale("log")
    ax.set_xlabel("training budget (timesteps)")
    ax.set_ylabel("crash-rate gap (A - B)")
    ax.set_title(f"Gap vs training budget ({condition})")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend(fontsize="small")


def plot_gap_vs_w(df, ax):
    """
    Bonus figure, in the spirit of the paper's own Figure 1: crash rate vs
    override weight w, one pair of lines (A vs B) for the smallest g and
    one pair for the largest g present in the data. Shows any erosion of
    the resistance directly, rather than only through the summary gap.
    """
    override_rows = df[df["condition"].str.startswith("override_w")].copy()
    if override_rows.empty:
        ax.set_title("No override_w rows found yet")
        return
    override_rows["w"] = override_rows["condition"].str.replace("override_w", "").astype(float)

    combos = override_rows[["network_size_scalar", "training_timesteps"]].drop_duplicates()
    combos = combos.sort_values(["network_size_scalar", "training_timesteps"])
    smallest, largest = combos.iloc[0], combos.iloc[-1]

    for label, combo in [("smallest g", smallest), ("largest g", largest)]:
        cell = override_rows[
            (override_rows["network_size_scalar"] == combo["network_size_scalar"]) &
            (override_rows["training_timesteps"] == combo["training_timesteps"])
        ]
        for arch, style in [("A", "--"), ("B", "-")]:
            sub = (cell[cell["architecture"] == arch]
                   .groupby("w")["crash_rate"].mean().reset_index().sort_values("w"))
            if sub.empty:
                continue
            ax.plot(sub["w"], sub["crash_rate"], style, marker="o",
                    label=f"{arch}, {label} ({combo['network_size_scalar']}, {int(combo['training_timesteps'])})")

    ax.set_xlabel("override / conflict weight (w)")
    ax.set_ylabel("crash rate")
    ax.set_title("Crash rate vs conflict weight, smallest vs largest g")
    ax.legend(fontsize="x-small")


def main(path):
    df = load(path)

    # The override condition is where the "resistance under pressure" claim
    # actually gets tested (in "standard", both sides should already be
    # low-crash, so a gap there isn't very informative), so it's the
    # default condition for the two required figures.
    condition = f"override_w{max(OVERRIDE_WEIGHTS)}"
    gap_df = compute_gap(df, condition)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_gap_vs_network_size(gap_df, condition, axes[0])
    plot_gap_vs_budget(gap_df, condition, axes[1])
    fig.tight_layout()
    fig.savefig("results/gap_vs_power.png", dpi=150)
    print("Saved results/gap_vs_power.png")

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    plot_gap_vs_w(df, ax2)
    fig2.tight_layout()
    fig2.savefig("results/crash_rate_vs_w.png", dpi=150)
    print("Saved results/crash_rate_vs_w.png")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else RESULTS_CSV_PATH
    main(csv_path)
