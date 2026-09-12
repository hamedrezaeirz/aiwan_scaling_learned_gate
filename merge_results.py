"""
merge_results.py
------------------
Merges every results/results_seeds_*.csv produced by a parallel
`--seeds` run (see launch_parallel.sh) into the canonical
results/results.csv that plot_results.py and the paper-update analysis
read from.

Usage (after all 4 launch_parallel.sh processes have finished):
    python3 merge_results.py
"""

import glob
import pandas as pd

from config import RESULTS_DIR, RESULTS_CSV_PATH

if __name__ == "__main__":
    files = sorted(glob.glob(f"{RESULTS_DIR}/results_seeds_*.csv"))
    if not files:
        raise SystemExit(
            f"No results_seeds_*.csv files found in {RESULTS_DIR}/ -- "
            f"nothing to merge yet."
        )
    print(f"Merging {len(files)} files:")
    for f in files:
        print(f"  {f}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nWrote {len(df)} total rows to {RESULTS_CSV_PATH}")
