"""Compare B_deficit with A, B (old, level-fed gate) and C from results.csv.
Run from aiwan_scaling/ after merge_B_deficit.py."""
import pandas as pd
from scipy import stats

old = pd.read_csv("results/results.csv")
new = pd.read_csv("results/results_B_deficit.csv")
df = pd.concat([old, new], ignore_index=True)
# restrict old rows to the cells/conditions B_deficit actually covers
keys = new[["network_size", "training_timesteps", "condition"]].drop_duplicates()
df = df.merge(keys, on=["network_size", "training_timesteps", "condition"])
print("cells covered:", new.groupby(["network_size", "training_timesteps"]).ngroups,
      "| seeds:", sorted(new.seed.unique()))

print("\n== mean crash rate by condition and architecture (over covered cells) ==")
print(df.pivot_table(index="condition", columns="architecture", values="crash_rate").round(3))

print("\n== override_w8: A - gate gap per cell (mean over seeds) ==")
w8 = df[df.condition == "override_w8"].pivot_table(
    index=["network_size", "training_timesteps"], columns="architecture", values="crash_rate")
for col in ("B", "B_deficit", "C"):
    if col in w8:
        g = (w8["A"] - w8[col])
        print(f"A-{col}: min {g.min():.3f}  max {g.max():.3f}  mean {g.mean():.3f}")
print("B_deficit crash: min %.3f max %.3f" % (w8.B_deficit.min(), w8.B_deficit.max()))

print("\n== does the A - B_deficit gap narrow with budget? (w=8, per-seed, sizes pooled) ==")
x = df[(df.condition == "override_w8") & df.architecture.isin(["A", "B_deficit"])]
p = x.pivot_table(index=["seed", "training_timesteps"], columns="architecture", values="crash_rate")
gap = (p["A"] - p["B_deficit"]).groupby(["seed", "training_timesteps"]).mean().unstack()
lo, hi = gap.columns.min(), gap.columns.max()
t = stats.ttest_rel(gap[lo], gap[hi])
print(f"gap@{lo}: {gap[lo].mean():.3f}  gap@{hi}: {gap[hi].mean():.3f}  "
      f"paired t({len(gap)-1})={t.statistic:.2f}, p={t.pvalue:.3g}")

print("\n== low-conflict check: B_deficit vs A at w=0 and w=1 ==")
for c in ("override_w0", "override_w1"):
    if c in set(df.condition):
        q = df[df.condition == c].pivot_table(columns="architecture", values="crash_rate")
        print(c, q.round(3).to_dict("records")[0])
