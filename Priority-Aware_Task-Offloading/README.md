# Priority-Aware Task Offloading: code, raw outputs and results

Companion repository for the journal paper **"Priority-Aware Task Offloading In The Internet of Medical Things Using Deep Reinforcement Learning"** (A. S. Seidu, J. O. Agyemang, K. Agyeman-Prempeh Agyekum, K. Opuni-Boachie Obour Agyekum, F. A. Acheampong, K. Adu-Boahen Opare; Kwame Nkrumah University of Science and Technology), submitted to *Intelligence-Based Medicine*. The experiments began as part of the first author's MPhil thesis (*SDN-Enabled Dueling Deep Q-Network Mechanism for Energy-Efficient Task Offloading in IoMT Applications*, KNUST); Sections 1-7 below keep the thesis numbering, and Sections 8-10 and the paper map in Section 5 cover the journal revision.

Everything here is simulation. Nothing in this repository is a clinical result or a clinical safety claim.

The repository holds the simulator, the agents and policies, the raw run outputs, the analysis scripts, the final result tables and the figures. It lets a reader (1) re-run the experiments, (2) regenerate every table and figure from the raw outputs, and (3) trace each number in the paper to a file.

**Quick start (no GPU, a few minutes):** `pip install -r requirements.txt`, then run the analysis chain in Section 4 from the repository root. `bash run_commands.sh`, `run_ctrlplane.sh` and the three Colab notebooks re-run the learned agents (several core-hours per configuration).

## 1. Contents

| Folder / file | Content |
|---|---|
| `code/iomt_env.py` | Event-driven simulator, state, action, task generation, `compute_reward` |
| `code/policies.py` | Non-learning policies: always-local, random, round-robin, greedy (least-loaded MEC), priority-aware feasible-min-energy rule (and a priority-blind version, not reported) |
| `code/agents_v2.py` | PyTorch DQN with switches for the dueling head and the double-estimator target |
| `code/run_v2.py` | Experiment runner (parallel, resumable); writes `per_seed.csv`, `convergence.csv`, `summary.csv`, `paired_tests.csv` per output folder |
| `raw/` | Raw outputs of every run (7 folders), exactly as produced by `run_v2.py` |
| `analysis/merge_raw.py` | Merges `raw/` into `results/all_per_seed.csv` and `results/all_conv.csv` |
| `analysis/summarise_and_test.py` | Means, t-based 95% CIs, paired Wilcoxon tests with Holm correction and bootstrap CIs (main and architecture families) |
| `analysis/ablation_tests.py` | Paired tests of the five ablation variants against the full agent |
| `analysis/convergence_stats.py` | Training-convergence statistics (Table 4.4) |
| `analysis/make_figures.py` | Figures 4.1-4.6 |
| `results/` | Final tables used in the thesis |
| `figures/` | Thesis figures 4.1-4.6, extension figures, and `figures/paper/` (paper Figures 2-4). The architecture figure (paper Fig. 1) is the authors' own drawing and is not included |
| `code/run_robust.py`, `code/ppo_agent.py`, `code/run_safety.py` | Robustness, PPO and Tier-1 safeguard runners (Sections 8-10) |
| `analysis/ctrlplane_analysis.py`, `robust_analysis.py`, `safety_analysis.py`, `make_paper_figures.py` | Analysis for Sections 8-10 and Figures 2-4 of the paper |
| `*_colab.ipynb`, `run_ctrlplane.sh` | Colab notebooks and script that produced the control-plane, robustness and safeguard runs |
| `run_commands.sh` | The exact commands for every raw folder and for the analysis chain |
| `requirements.txt` | Software versions |
| `SHA256SUMS.txt` | Checksums of all files |

## 2. Experimental design (summary of Chapter 3)

- **System:** wearable devices (N = 20 baseline; 40 and 60 for scaling), 3 MEC servers (5-10 GHz, other-tenant share following a bounded random walk around 0.25 in [0.05, 0.5], FIFO work backlog), a cloud (50 GHz, 20-80 ms wide-area delay, no queue). Wireless rate from the Shannon capacity over 10 MHz with SNR drawn uniformly between 5 and 30 dB per link and decision. Device CPU 0.5-1.0 GHz, transmit power 0.1-0.5 W, battery 2,664 J, switched capacitance 1e-28.
- **Tasks:** Poisson arrivals at 6N tasks/s. Priority tier mix 5/15/80% (Tier 1/2/3) with deadlines 50/100/5,000 ms. Task type light/medium/heavy with probabilities 0.3/0.4/0.3, independent of tier.
- **MDP:** 13-dimensional state (B, f, Q, D, C, P, R, L1-L3, SNR1-SNR3), 5 actions (local, MEC 1-3, cloud). Reward = -(alpha * w * min(T/Tmax, 3) + beta * min(E/0.02, 1) + lambda * 1[T > Tmax]) with alpha = 0.6, beta = 0.4, lambda = 50, w = 4 - P.
- **Agents:** two shared hidden layers of 128 ReLU units, 64-unit value and advantage streams for the dueling networks; Adam, learning rate 1e-3, gamma = 0.95, replay 10,000, batch 64, target update every 100 gradient steps, gradient-norm clip 1.0; epsilon decays geometrically from 1.0 to 0.01 over 20,000 decisions.
- **Protocol:** 500 episodes of 100 decisions (environment not reset between episodes); greedy evaluation on 2,000 decisions in a new ward (environment seed 10000 + training seed); 10 seeds (0-9); common random numbers across methods for the same (N, seed).
- **Methods:** learned: `dueling_ddqn` (proposed, Dueling Double DQN / D3QN), `standard_dqn`, `double_dqn`, `dueling_dqn`. Policies: `always_local`, `random`, `round_robin`, `greedy`, `heuristic_pa` (priority-aware rule). Ablations: `no_priority_reward`, `no_priority_state`, `tier_blind`, `no_sdn` (thesis label "No per-server state": 7-D state B, f, Q, D, C, P, R, which keeps the mean link rate R), `no_sdn_loads` (10-D state without L1-L3).
- **Statistics:** mean and t-based 95% CI (9 degrees of freedom); exact two-sided Wilcoxon signed-rank test on paired per-seed differences (comparison arm minus proposed agent); Holm correction within each family and (N, metric): main = 6 comparisons, architectures = 3, ablation = 5; bootstrap 95% CI of the mean difference (5,000 resamples, generator seed 0). With 10 pairs the smallest attainable p is 0.002.

## 3. Which raw folder holds what

| Folder | Methods | N | Seeds |
|---|---|---|---|
| `raw/ablation_lab_N20`, `raw/ablation_lab_N60` | dueling_ddqn, no_priority_reward, no_priority_state, tier_blind, no_sdn, no_sdn_loads, heuristic_pa | 20 / 60 | 0-9 |
| `raw/nonlearning_baselines` | always_local, greedy, heuristic_pa, random, round_robin | 20, 60 | 0-9 |
| `raw/main_A` | standard_dqn | 20, 40, 60 | 0-9 |
| `raw/main_B1` | dueling_ddqn | 40 | 0-9 |
| `raw/main_B2` | greedy, round_robin, random, always_local, heuristic_pa | 40 | 0-9 |
| `raw/arch_C` | dueling_dqn, double_dqn | 20, 60 | 0-9 |

`heuristic_pa` appears in two folders at N = 20 and 60. It is deterministic given the seed, and `merge_raw.py` keeps the first copy. Total: 350 method x N x seed rows.

## 4. Reproducing the results

```
pip install -r requirements.txt
# Option A: use the supplied raw outputs (minutes)
python analysis/merge_raw.py
python analysis/summarise_and_test.py
python analysis/convergence_stats.py
python analysis/ablation_tests.py
python analysis/make_figures.py
python analysis/ctrlplane_analysis.py && python analysis/robust_analysis.py && python analysis/safety_analysis.py && python analysis/make_paper_figures.py
# Option B: re-run the experiments (see run_commands.sh; the learned agents take several core-hours per configuration)
```

Checks made when the package was assembled:
- `merge_raw.py` reproduces `results/all_per_seed.csv` and `results/all_conv.csv` exactly from `raw/`.
- The statistics scripts reproduce `summary_main.csv`, `paired_main.csv`, `conv_stats.csv` and `paired_ablation.csv` (differences below 1e-14, which is floating-point rounding).
- `make_figures.py` reproduces the six figures pixel for pixel.
- Re-running the non-learning policies (greedy, round-robin, priority-aware rule, N = 20, seeds 0-1) with `run_v2.py` gives values identical to the stored ones.
- The learned agents were not re-run when the package was assembled. Their results depend on the PyTorch version, the CPU and the thread count, so a re-run is expected to agree statistically but not bit for bit.

## 5. Where each thesis table and figure comes from

| Thesis item | Source |
|---|---|
| Table 4.1 energy, N = 20, and paired differences | `summary_main.csv` (`energy_mean`), `paired_main.csv` (family `main`, metric `energy_mean`) |
| Tables 4.2, 4.3 time and violations by tier, N = 20 | `summary_main.csv` (`time_t*_mean`, `viol_*`) |
| Table 4.4 convergence | `conv_stats.csv` |
| Section 4.5.1 decision time | `summary_main.csv` (`infer_ms`, per-seed in `all_per_seed.csv`) |
| Tables 4.5, 4.6 architectures | `summary_main.csv` (four learned arms), `paired_main.csv` (family `arch`) |
| Tables 4.7, 4.8 scaling | `summary_main.csv` (N = 20, 40, 60) |
| Tables 4.9, 4.10 ablation | `summary_main.csv`, `paired_ablation.csv` |
| Table 4.11 related work | qualitative, no data file |
| Figures 4.1-4.6 | `figures/`, from `analysis/make_figures.py` |
| Training times (Table 4.4) | `all_per_seed.csv` (`train_time_s`); these depend on batch size and machine load and are only indicative |

Column dictionary for `all_per_seed.csv`: `energy_mean` (J per task), `time_mean`, `time_t1_mean` ... (ms), `viol_overall`, `viol_t1` ... (%), `eval_reward` (canonical full reward, identical for every arm), `t1_p99_ms`, `t1_n`, `t1_viol_n` (Tier-1 task counts), `act_t{tier}_a{action}` (action shares by tier), `train_time_s`, `infer_ms` (ms per decision).

## 5a. Where each paper table and figure comes from

| Paper item | Source |
|---|---|
| Table 4 (N = 20) | `results/summary_main.csv` (`energy_mean`, `time_mean`, `time_t1_mean`, `viol_t1`, `viol_overall`), tests in `paired_main.csv` |
| Table 5 (network size) | `summary_main.csv`, `viol_t1` at N = 20, 40, 60 |
| Table 6 (ablation) | `summary_main.csv`, `paired_ablation.csv` |
| Table 7, Fig. 3 (control-plane cost) | `ctrlplane_summary.csv`, `ctrlplane_device_local_vs_ref.csv`, `ctrlplane_delta_vs_none.csv`; `figures/paper/ctrl.png` |
| Table 8, Fig. 4, Section 4.7 (PPO, one-step reference, unseen conditions) | `robust_summary.csv`, `robust_paired_vs_ref.csv`, `robust_shift_vs_default.csv`; `figures/paper/robust.png` |
| Table 10 and Section 4.10 (Tier-1 safeguard) | `safety_summary.csv`, `safety_paired_vs_unprotected.csv`, `figures/Fig_safety_seeds.png` |
| Fig. 2 (learning curves) | `figures/paper/learning.png` (copy of `figures/Fig4.4_training_reward.png`), `all_conv.csv` |
| Tables 1, 3, 9 | Qualitative or design values; Table 9 numbers are taken from the cited papers, not from this repository |

## 6. Known limitations to keep in mind

- All results come from simulation with author-chosen parameters, deadlines and thresholds. The SDN control plane is idealised.
- The `no_sdn` ablation keeps the mean link rate R, so it is a reduced network view and not a strictly device-local one.
- `run_v2.py` lists `heuristic_np` (priority-blind rule) among its default `main` arms; it is not part of the reported results. The ablation default list omits `tier_blind` and `no_sdn_loads`, which were run with explicit `--arms`. `run_commands.sh` states the explicit arms used.
- Training and decision times were measured on shared machines and in parallel jobs, so they are not benchmarks.

## 7. Provenance and software

- Raw runs: the first run set (`raw/main_*`, `raw/arch_C`, `raw/ablation_lab_*`, `raw/nonlearning_baselines`) was produced on the authors' lab computers with parallel CPU jobs; the control-plane, robustness and safeguard learned runs were produced on Google Colab (CPU, 2 parallel jobs, one thread each; logs in `logs/`). Learned results depend on the PyTorch version and CPU, so a re-run is expected to agree statistically, not bit for bit.
- All analysis scripts were re-run from the raw files for this release and reproduce every table in `results/` (differences at most 1e-7, confidence-interval widths only, from scipy-version rounding) with Python 3.13, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5 and matplotlib 3.11.2. PyTorch is needed only to re-run the learned agents.
- `SHA256SUMS.txt` lists the checksum of every file (`sha256sum -c SHA256SUMS.txt`).
- Licence: MIT (`LICENSE`). To cite: see `CITATION.cff`.

## 8. Control-plane cost extension (added after the first journal review)

Reviewer comments asked for the cost of controller queries, signalling and flow-rule installation. `code/iomt_env.py` and `code/run_v2.py` were extended; the unmodified files used for all results in Sections 1-7 are kept in `code/original_as_run/`. With the new options left at their defaults (`--ctrl none`) the extended code reproduces the earlier results (checked: non-learning policies, N = 20 and 60, seeds 0-1, maximum difference 2e-13).

New simulator options (all zero by default):

| Option | Meaning |
|---|---|
| `ctrl_query_s` | Delay (s) of one state query to the SDN controller, paid before the decision by every method that uses controller-supplied state (all DQN agents except `device_local`, `greedy`, `heuristic_pa`). Not paid by `always_local`, `random`, `round_robin`, `device_local`. |
| `ctrl_msg_bits` | Query message size (bits). Device energy of one query = transmit power x bits / mean uplink rate. |
| `flow_s` | Reactive flow-rule installation delay (s), paid by every task sent to a MEC server or the cloud, by every method. |

Presets (`--ctrl`): `none` 0/0 ms; `low` query 2 ms, flow 5 ms; `mid` 5/10 ms; `high` 10/20 ms; query message 500 bytes (4,000 bits). These values are assumptions for a sensitivity study, not measurements. The delays are added to the task completion time and the signalling energy to the device energy, so they enter the reward, and the learning agents are retrained in each scenario. A new arm `device_local` (6-D state B, f, Q, D, C, P; no link rate, no server loads, no controller query) is the strictly device-local ablation.

Run `bash run_ctrlplane.sh` (set `JOBS`); then `python analysis/ctrlplane_analysis.py` writes `results/ctrlplane_*.csv` and `figures/Fig_ctrlplane_*.png`. Raw folders: `raw/ctrl_{low,mid,high}_nonlearning` (five non-learning policies, N = 20 and 60, 10 seeds), `raw/ctrl_{low,mid,high}_learned` (`dueling_ddqn`, retrained in each scenario, N = 20 and 60, 10 seeds) and `raw/ctrl_{none,low,mid,high}_device_local` (`device_local`, same protocol). The `none` scenario for all other methods is the original run set. The learned runs were executed on Google Colab (CPU, 2 parallel jobs, one thread each; logs in `logs/`); training times are therefore not comparable with the first run set.

Result files written by `analysis/ctrlplane_analysis.py`: `results/ctrlplane_all_per_seed.csv` (merged, 4 scenarios), `ctrlplane_summary.csv` (mean and 95% CI), `ctrlplane_paired_vs_ref.csv` (each method minus the proposed agent within a scenario; exact Wilcoxon, Holm over the comparison methods), `ctrlplane_delta_vs_none.csv` (each method's change relative to the ideal controller, paired by seed, Holm over the three scenarios), `ctrlplane_device_local_vs_ref.csv` (device-local agent minus the proposed agent, paired by seed, Holm over the four scenarios), and the figures `Fig_ctrlplane_tier1.png` and `Fig_ctrlplane_energy.png`. Duplicate rows caused by a restarted Colab job were removed (first copy kept) before analysis.

## 9. Robustness and extra-baseline experiments (R2.6 and R2.5 of the first review)

New files: `code/run_robust.py` (runner), `code/ppo_agent.py` (PPO), `analysis/robust_analysis.py`. `code/iomt_env.py` gained three options and a read-only look-ahead function; with the options at their defaults the environment reproduces the previous version exactly (checked: identical state and outcome sequences over 3,000 random decisions, with and without control-plane cost; `peek` equals the outcome of `step` for every action).

New simulator options (off by default): `burst=(hi, lo, dwell_s)` two-state arrival rate multiplier (own random stream; 1.6/0.4/1 s keeps the mean rate); `cap_drop=(after_decisions, factor)` every MEC server's CPU multiplied after that many decisions; `snr_range=(lo, hi)` link SNR range in dB; `peek(action)` returns the delay and energy an action would give for the current task without changing the environment.

Protocol. Agents are trained once in the default environment (ideal controller, 500 x 100 decisions, seeds 0-9) and tested without re-training in 11 environments (2,000 greedy decisions each, environment seed 10000 + seed): `default`; the other sizes among N = 20, 40, 60, 80, 100 (three MEC servers in every case, so N >= 80 is a stress test of saturation); and at the training size `burst` (arrival-rate multiplier 1.6/0.4, mean dwell 1 s), `tier_shift` (Tier mix 15/25/60 %), `heavy_tasks` (light/medium/heavy 10/30/60 %), `poor_channel` (SNR 0-20 dB), `cap_drop` (-50 % MEC CPU after 1,000 decisions) and `combined` (all four). Learned methods: `dueling_ddqn` (trained at N = 20, 60) and `ppo` (N = 20, 40, 60). Non-learning: the five policies of Section 1 and `oracle_onestep`, a reference that sees the true delay and energy of every action for the current task and selects the one with the best reward (no look-ahead beyond the current task; not deployable, it shows the headroom).

PPO details: separate actor and critic MLPs (2 x 128 ReLU), rollout 128 decisions, 8 epochs, minibatch 32, learning rate 3e-4, gamma 0.95, GAE lambda 0.95, clip 0.2, entropy coefficient 0.01, value coefficient 0.5, gradient clipping 1.0, rewards scaled by 0.1; same 50,000-decision budget as the DQN agents. No hyper-parameters were tuned for any learned method.

Raw folders: `raw/robust_nonlearning`, `raw/robust_dueling`, `raw/robust_ppo` (one row per method x training size x test environment x seed). Outputs of `python analysis/robust_analysis.py`: `results/robust_all_per_seed.csv`, `robust_summary.csv`, `robust_paired_vs_ref.csv` (each method minus the proposed agent, exact Wilcoxon, Holm over the comparison methods), `robust_shift_vs_default.csv` (change of each method relative to its own default-environment result, paired by seed, Holm over the test environments), and the figures `Fig_robust_sizes.png`, `Fig_robust_workloads.png`, `Fig_baselines_default.png`. Colab notebook: `robust_colab.ipynb`.

## 10. Tier-1 safety shield experiment (code/run_safety.py, analysis/safety_analysis.py)
A dueling D3QN agent (trained as in `run_robust.py`, N_train = 20 and 60, seeds 0-9) is evaluated without further training in the
robustness test environments under four variants: `unprotected`; `shield_0.8` / `shield_0.5` (a Tier-1 action whose nominal estimated
completion time exceeds 0.8 / 0.5 x deadline is replaced by the destination with the smallest estimate); `t1_rule` (Tier-1 tasks follow the
priority-aware rule). Estimates use only state-vector quantities and nominal constants (never the true outcome). Notebook: `safety_colab.ipynb`.
Command: `cd code && python run_safety.py --N-train 20,60 --seeds 10 --jobs 2 --out ../raw/safety` then `python analysis/safety_analysis.py`.
Results are in `raw/safety`, `results/safety_*.csv`, `figures/Fig_safety_seeds.png`. Simulation results only; they say nothing about clinical safety.
