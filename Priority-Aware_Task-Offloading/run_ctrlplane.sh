#!/usr/bin/env bash
# Control-plane cost experiment (reviewer comments R2.2 and R3-M2). Run from the package root: bash run_ctrlplane.sh
# Set JOBS to the number of CPU cores you can use (each learning run uses one thread, about 300 s).
# Every command is resumable: if it stops, run it again and it continues from the last finished job.
JOBS=${JOBS:-4}
cd "$(dirname "$0")/code"

# --- A. NON-LEARNING policies under the three control-plane scenarios (fast, deterministic; already supplied in raw/) ---
for sc in low mid high; do
  python run_v2.py --suite main --arms always_local,random,round_robin,greedy,heuristic_pa --N 20,60 --seeds 10 --jobs $JOBS --ctrl $sc --out ../raw/ctrl_${sc}_nonlearning
done

# --- B. REQUIRED: proposed agent (dueling_ddqn), retrained and tested in each scenario. 3 scenarios x 2 N x 10 seeds = 60 runs ---
for sc in low mid high; do
  python run_v2.py --suite main --arms dueling_ddqn --N 20,60 --seeds 10 --jobs $JOBS --ctrl $sc --out ../raw/ctrl_${sc}_learned
done

# --- C. RECOMMENDED: strictly device-local agent (6-D state, no controller query). It also answers R3-M2 (device-local ablation).
#        C1 is the ideal-controller run of the original protocol (20 runs); C2 is the three scenarios (60 runs). ---
python run_v2.py --suite main --arms device_local --N 20,60 --seeds 10 --jobs $JOBS --ctrl none --out ../raw/ctrl_none_device_local
for sc in low mid high; do
  python run_v2.py --suite main --arms device_local --N 20,60 --seeds 10 --jobs $JOBS --ctrl $sc --out ../raw/ctrl_${sc}_device_local
done

# --- D. OPTIONAL: Standard DQN in the three scenarios (60 runs), to check that the architecture conclusion holds under control-plane cost ---
# for sc in low mid high; do
#   python run_v2.py --suite main --arms standard_dqn --N 20,60 --seeds 10 --jobs $JOBS --ctrl $sc --out ../raw/ctrl_${sc}_standard
# done

# --- Analysis (run from the package root) ---
cd ..
python analysis/ctrlplane_analysis.py
