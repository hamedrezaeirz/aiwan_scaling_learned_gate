#!/bin/bash
# launch_parallel.sh
# -------------------
# Splits config.SEEDS = [0..9] across 4 processes (one per core), so the
# full sweep fits inside the VPS's remaining time window. Each process
# writes its own results/results_seeds_<...>.csv (see run_experiment.py),
# so there's no write conflict between them -- merge afterwards with
# merge_results.py.
#
# Estimated: ~54 hours (~2.2 days) total on 4 cores for the full sweep as
# currently configured (config.py). Run this inside tmux or screen, or
# with nohup, so it survives an SSH disconnect:
#
#   tmux new -s aiwan
#   bash launch_parallel.sh
#   [Ctrl+b, d to detach -- reattach later with: tmux attach -t aiwan]
#
# or:
#
#   nohup bash launch_parallel.sh > launch.log 2>&1 &
#
# Check progress any time with:
#   tail -f logs/seeds_0_1_2.log     (etc. for the other 3 log files)
#   wc -l results/results_seeds_*.csv   (row counts so far, per process)

set -e
cd "$(dirname "$0")"
mkdir -p results logs

echo "Starting 4 parallel processes across seeds 0-9..."

python3 run_experiment.py --seeds 0 1 2 > logs/seeds_0_1_2.log 2>&1 &
PID1=$!
python3 run_experiment.py --seeds 3 4 5 > logs/seeds_3_4_5.log 2>&1 &
PID2=$!
python3 run_experiment.py --seeds 6 7   > logs/seeds_6_7.log   2>&1 &
PID3=$!
python3 run_experiment.py --seeds 8 9   > logs/seeds_8_9.log   2>&1 &
PID4=$!

echo "Launched PIDs: $PID1 $PID2 $PID3 $PID4"
echo "Waiting for all 4 to finish (this is the ~2.2 day part)..."

wait $PID1 $PID2 $PID3 $PID4

echo "All 4 processes finished. Run: python3 merge_results.py"
