"""Merge results/results_B_deficit_seeds_*.csv -> results/results_B_deficit.csv
(never touches results.csv)."""
import glob
import pandas as pd
from config import RESULTS_DIR

files = sorted(glob.glob(f"{RESULTS_DIR}/results_B_deficit_seeds_*.csv"))
if not files:
    raise SystemExit("no results_B_deficit_seeds_*.csv found")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
dup = df.duplicated(["network_size", "training_timesteps", "seed", "condition"]).sum()
if dup:
    raise SystemExit(f"{dup} duplicate rows (a seed was run twice?) -- fix before merging")
df.to_csv(f"{RESULTS_DIR}/results_B_deficit.csv", index=False)
print(f"wrote {len(df)} rows from {len(files)} files")
