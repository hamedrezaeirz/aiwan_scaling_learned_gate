#!/bin/bash
# Usage: bash launch_B_deficit.sh [--reduced]      (run from aiwan_scaling/)
# 4 processes over seeds 0-9, same split as launch_parallel.sh. Then:
#   python3 merge_B_deficit.py && python3 analyze_B_deficit.py
set -e
cd "$(dirname "$0")"; mkdir -p results logs
MODE="$1"
python3 run_experiment_B_deficit.py $MODE --seeds 0 1 2 > logs/Bdef_0_1_2.log 2>&1 &
python3 run_experiment_B_deficit.py $MODE --seeds 3 4 5 > logs/Bdef_3_4_5.log 2>&1 &
python3 run_experiment_B_deficit.py $MODE --seeds 6 7   > logs/Bdef_6_7.log   2>&1 &
python3 run_experiment_B_deficit.py $MODE --seeds 8 9   > logs/Bdef_8_9.log   2>&1 &
wait
echo "done -- now: python3 merge_B_deficit.py && python3 analyze_B_deficit.py"
