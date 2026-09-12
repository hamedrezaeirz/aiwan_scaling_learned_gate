"""
plot_gate_results.py
---------------------
Companion to plot_results.py, producing the two figures that report
Architecture C's (learned-gate) headline result -- the crash-rate gap
against Architecture A collapsing with training budget, which is what
AIWAN §7.3 reports as confirming the §4.2 collapse prediction.

Deduplicates by logical key first: an early relaunch (documented in
project history) left cell (net_size=16, budget=10000) written twice
for seeds 0, 3, 6, 8. Row COUNTS alone can look complete even when a
duplicate is masking a missing cell elsewhere, so this also asserts the
full grid (10 seeds x 24 cells) is actually present, not just that the
row count matches.

Usage:
    python3 plot_gate_results.py [path_to_csv]
    (defaults to config.RESULTS_CSV_PATH)

Requires: pip install pandas matplotlib scipy
"""

import sys
import ast
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from config import RESULTS_CSV_PATH, NETWORK_SIZES, TRAINING_BUDGETS, SEEDS


def load_deduped(path):
    df = pd.read_csv(path)
    df["net"] = df["network_size"].apply(lambda x: ast.literal_eval(x)[0])

    key_cols = ["architecture", "net", "training_timesteps", "seed", "condition"]
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="first")
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate rows (same architecture/net/budget/"
              f"seed/condition key, kept the first).")

    expected_cells = set(itertools.product([n[0] for n in NETWORK_SIZES], TRAINING_BUDGETS))
    missing = []
    for seed in SEEDS:
        present = set(df[df.seed == seed].groupby(["net", "training_timesteps"]).size().index)
        gap = expected_cells - present
        if gap:
            missing.append((seed, gap))
    if missing:
        raise SystemExit(f"Incomplete data -- missing cells: {missing}. "
                          f"Fix before generating figures; a duplicate elsewhere "
                          f"can make the total row count look fine while a real "
                          f"cell is silently absent.")
    print(f"Verified complete: {len(SEEDS)} seeds x {len(expected_cells)} cells, "
          f"{len(df)} rows.")
    return df


def figure_gap_vs_power(df, condition, out_path):
    """Gap (A - C) vs network size and training budget, mirroring
    plot_results.py's gap_vs_power.png but for the learned gate."""
    sub = df[df.condition == condition]
    means = sub.groupby(["architecture", "net", "training_timesteps"])["crash_rate"].mean()
    stds = sub.groupby(["architecture", "net", "training_timesteps"])["crash_rate"].std()
    n = len(SEEDS)

    a_mean = means.xs("A", level="architecture")
    c_mean = means.xs("C", level="architecture")
    a_std = stds.xs("A", level="architecture")
    c_std = stds.xs("C", level="architecture")

    gap_mean = (a_mean - c_mean).rename("gap_mean")
    gap_se = (np.sqrt(a_std**2 + c_std**2) / np.sqrt(n)).rename("gap_se")
    gap_df = pd.concat([gap_mean, gap_se], axis=1).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for budget, g in gap_df.groupby("training_timesteps"):
        g = g.sort_values("net")
        axes[0].errorbar(g["net"], g["gap_mean"], yerr=g["gap_se"], marker="o",
                          label=f"budget={budget}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("network size (hidden units)")
    axes[0].set_ylabel("crash-rate gap (A - C)")
    axes[0].set_title(f"Gap (A-C) vs network size ({condition})")
    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].legend(fontsize="small")

    for net, g in gap_df.groupby("net"):
        g = g.sort_values("training_timesteps")
        axes[1].errorbar(g["training_timesteps"], g["gap_mean"], yerr=g["gap_se"],
                          marker="o", label=f"net_size={net}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("training budget (timesteps)")
    axes[1].set_ylabel("crash-rate gap (A - C)")
    axes[1].set_title(f"Gap (A-C) vs training budget ({condition})")
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].legend(fontsize="small")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def figure_crash_vs_w_largest(df, out_path):
    """Crash rate vs conflict weight w, A vs B vs C, at the largest
    (network_size, budget) cell tested."""
    override = df[df.condition.str.startswith("override_w")].copy()
    override["w"] = override["condition"].str.replace("override_w", "").astype(float)

    largest_net = max(n[0] for n in NETWORK_SIZES)
    largest_budget = max(TRAINING_BUDGETS)
    cell = override[(override.net == largest_net) & (override.training_timesteps == largest_budget)]

    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {"A": ("--", "#d64545", "A (single-loop)"),
              "B": ("-.", "#2f6fab", "B (hardcoded gate)"),
              "C": ("-", "#3ba55d", "C (learned gate)")}
    for arch, (ls, color, label) in styles.items():
        s = cell[cell.architecture == arch].groupby("w")["crash_rate"].mean().reset_index().sort_values("w")
        ax.plot(s["w"], s["crash_rate"], ls, marker="o", color=color, label=label)

    ax.set_xlabel("override / conflict weight (w)")
    ax.set_ylabel("crash rate")
    ax.set_title(f"Crash rate vs conflict weight, largest g tested\n"
                 f"(net_size={largest_net}, budget={largest_budget})")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def report_stats(df):
    """Prints the paired t-tests reported in AIWAN §7.3, computed
    directly from this results.csv, so the numbers in the paper are
    reproducible from this exact script."""
    ow8 = df[df.condition == "override_w8"]

    def paired_gap_test(group_col, lo_val, hi_val):
        lo = ow8[(ow8[group_col] == lo_val) & (ow8.architecture.isin(["A", "C"]))]
        hi = ow8[(ow8[group_col] == hi_val) & (ow8.architecture.isin(["A", "C"]))]
        lo_p = lo.pivot_table(index="seed", columns="architecture", values="crash_rate")
        hi_p = hi.pivot_table(index="seed", columns="architecture", values="crash_rate")
        lo_gap = lo_p["A"] - lo_p["C"]
        hi_gap = hi_p["A"] - hi_p["C"]
        t, p = stats.ttest_rel(lo_gap, hi_gap)
        return lo_gap.mean(), hi_gap.mean(), t, p

    lo_g, hi_g, t, p = paired_gap_test("training_timesteps", 10000, 500000)
    print(f"\nBudget 10k vs 500k gap (A-C), override_w8: {lo_g:.4f} -> {hi_g:.4f}  "
          f"paired t(9)={t:.2f}, p={p:.4g}")

    lo_g, hi_g, t, p = paired_gap_test("net", 16, 2048)
    print(f"Net 16 vs 2048 gap (A-C), override_w8: {lo_g:.4f} -> {hi_g:.4f}  "
          f"paired t(9)={t:.2f}, p={p:.4g}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else RESULTS_CSV_PATH
    df = load_deduped(csv_path)
    figure_gap_vs_power(df, "override_w8", "results/gap_AC_vs_power.png")
    figure_crash_vs_w_largest(df, "results/crash_rate_vs_w_ABC_largest.png")
    report_stats(df)
