"""
plot_results_B_deficit.py
Regenerates Figures 7 and 8 of the AIWAN paper with BOTH versions of Architecture B
(as originally run, level-fed gate; and B_deficit, gate direction corrected) and with
+-1 SE over seeds (SD / sqrt(n_seeds)), matching Figure 5.
Run from aiwan_scaling/ after merge_B_deficit.py:
    python3 plot_results_B_deficit.py
Writes results/gap_vs_power_B_deficit.png and results/crash_rate_vs_w_B_deficit.png.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

old = pd.read_csv("results/results.csv")
new = pd.read_csv("results/results_B_deficit.csv")
df = pd.concat([old, new], ignore_index=True)
df["net"] = df["network_size"].str.extract(r"(\d+)").astype(int)

def gap_table(arch_b):
    p = (df[(df.condition == "override_w8") & df.architecture.isin(["A", arch_b])]
         .pivot_table(index=["net", "training_timesteps", "seed"], columns="architecture", values="crash_rate"))
    g = (p["A"] - p[arch_b]).rename("gap").reset_index()
    n = g.seed.nunique()
    out = g.groupby(["net", "training_timesteps"])["gap"].agg(["mean", "std"]).reset_index()
    out["se"] = out["std"] / np.sqrt(n)
    return out

def fig7():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    rows = [("B", "B as originally run (gate fed the battery level)"),
            ("B_deficit", "B with the gate direction corrected")]
    for r, (arch, label) in enumerate(rows):
        t = gap_table(arch)
        for budget, sub in t.groupby("training_timesteps"):
            axes[r, 0].errorbar(sub["net"], sub["mean"], yerr=sub["se"], marker="o", capsize=2, label=f"budget={budget}")
        for net, sub in t.groupby("net"):
            axes[r, 1].errorbar(sub["training_timesteps"], sub["mean"], yerr=sub["se"], marker="o", capsize=2, label=f"net_size={net}")
        axes[r, 0].set_xscale("log"); axes[r, 1].set_xscale("log")
        axes[r, 0].set_xlabel("network size (hidden units)"); axes[r, 1].set_xlabel("training budget (timesteps)")
        axes[r, 0].set_title(f"Gap vs network size (override_w8)\n{label}", fontsize=10)
        axes[r, 1].set_title(f"Gap vs training budget (override_w8)\n{label}", fontsize=10)
        for c in range(2):
            axes[r, c].axhline(0, color="gray", lw=1)
            axes[r, c].set_ylim(-0.05, 0.85)
            axes[r, c].set_ylabel("crash-rate gap (A - B)")
            axes[r, c].legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig("results/gap_vs_power_B_deficit.png", dpi=150)
    print("saved results/gap_vs_power_B_deficit.png", fig.get_size_inches() * 150)

def fig8():
    ws = [0, 1, 1.5, 2, 4, 8]
    cells = [(16, 10000, "smallest g (16, 10000)"), (2048, 500000, "largest g (2048, 500000)")]
    styles = {"A": ("--", "o"), "B": ("-", "o"), "B_deficit": (":", "s")}
    names = {"A": "A", "B": "B (as originally run)", "B_deficit": "B (direction corrected)"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for net, bud, lab in cells:
        for arch in ["A", "B", "B_deficit"]:
            y = []
            for w in ws:
                c = f"override_w{w}"
                s = df[(df.architecture == arch) & (df.net == net) & (df.training_timesteps == bud) & (df.condition == c)]
                y.append(s.crash_rate.mean())
            ls, mk = styles[arch]
            ax.plot(ws, y, ls=ls, marker=mk, label=f"{names[arch]}, {lab}")
    ax.set_xlabel("override / conflict weight (w)"); ax.set_ylabel("crash rate")
    ax.set_title("Crash rate vs conflict weight, smallest vs largest g")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig("results/crash_rate_vs_w_B_deficit.png", dpi=150)
    print("saved results/crash_rate_vs_w_B_deficit.png", fig.get_size_inches() * 150)

if __name__ == "__main__":
    fig7(); fig8()
