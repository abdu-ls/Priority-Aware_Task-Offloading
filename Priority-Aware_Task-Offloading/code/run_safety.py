"""run_safety.py -- Tier-1 safety shield around a trained D3QN agent (reviewer comment R3-M4).

For each (N_train, seed) one dueling_ddqn agent is trained exactly as in run_robust.py and then evaluated, WITHOUT further
training, in the same test environments as the robustness study, under four variants that all use the same trained agent:
  unprotected   the agent's greedy action
  shield_0.8    for Tier-1 tasks, if the NOMINAL estimated completion time of the agent's action exceeds 0.8 x deadline,
                replace it by the destination with the smallest estimated completion time
  shield_0.5    same with 0.5 x deadline (more conservative)
  t1_rule       Tier-1 tasks always follow the priority-aware rule (FeasibleMinEnergy); other tiers follow the agent
The estimate uses only quantities visible in the state vector and the nominal constants in policies.py (7.5 GHz MEC, 0.3 W,
50 ms WAN), never the true outcome, so the shield is deployable. It does not include controller query/flow delays (evaluated in
the default 'none' control-plane setting only).
Output: one row per (N_train, seed, variant, test_env); extra column t1_override = share of Tier-1 decisions overridden.
--stub-policy lets you test the code path without torch (round_robin / random base policy instead of the agent).
"""
import argparse, os, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from iomt_env import IoMTEnv, compute_reward, KAPPA, CLOUD_CPU_HZ, BANDWIDTH_HZ, C_MAX, D_MAX, Q_REF, DEADLINE
from policies import F_MEC_NOM, PTX_NOM, WAN_NOM, FeasibleMinEnergy, RoundRobin, RandomPolicy
import run_v2 as R
from run_robust import test_envs

VARIANTS = {"unprotected": None, "shield_0.8": 0.8, "shield_0.5": 0.5, "t1_rule": "rule"}
_RULE = FeasibleMinEnergy(use_priority=True, use_global=True, margin=0.8)


def estimates(s):
    f, Qs = s[1] * 1e9, s[2] * Q_REF
    D, C = s[3] * D_MAX, s[4] * C_MAX
    load, snr = np.minimum(s[7:10], 0.95), s[10:13] * 30
    rate = BANDWIDTH_HZ * np.log2(1 + 10 ** (snr / 10))
    t = [Qs + C / f]
    for j in range(3):
        t.append(D / rate[j] + C / (F_MEC_NOM * (1 - load[j])))
    t.append(D / rate.max() + WAN_NOM + C / CLOUD_CPU_HZ)
    return np.array(t)


def protect(variant, act, s):
    """returns (action, overridden?)"""
    P = int(round(s[5] * 3)); v = VARIANTS[variant]
    if v is None or P != 1: return act, False
    if v == "rule":
        a = _RULE.act(s); return a, a != act
    t = estimates(s)
    if t[act] > v * DEADLINE[1]:
        a = int(np.argmin(t)); return a, a != act
    return act, False


def evaluate(base_act, variant, N, seed, a, kw):
    env = IoMTEnv(N, seed=R.EVAL_OFFSET + seed, arrival_rate=a["arrival_rate"], **kw); s = env.reset()
    rows, ov, n1 = [], 0, 0
    for _ in range(a["n_test"]):
        act0 = base_act(s); act, o = protect(variant, act0, s)
        if int(round(s[5] * 3)) == 1: n1 += 1; ov += int(o)
        s, info = env.step(act, queries=True)
        rows.append(dict(priority=info["priority"], energy=info["energy"], time_ms=info["time"] * 1e3,
                         violated=info["violated"], action=info["action"], reward=compute_reward(info)))
    return pd.DataFrame(rows), ov / max(n1, 1)


def run_job(n_train, seed, a):
    np.random.seed(seed); tt = np.nan
    if a["stub"]:
        pol = RoundRobin() if a["stub"] == "round_robin" else RandomPolicy(seed)
        base = lambda s: pol.act(s)
    else:
        agent, _, tt = R.train_drl(R.ARMS["dueling_ddqn"], n_train, seed, a); keep = R.ARMS["dueling_ddqn"]["keep"]
        base = lambda s: agent.act(s[keep], greedy=True)
    rows = []
    for name, N, kw in test_envs(n_train):
        for var in VARIANTS:
            if a["stub"] == "round_robin": pol.i = 0
            df, ovr = evaluate(base, var, N, seed, a, kw)
            r = dict(variant=var, N_train=n_train, test_env=name, test_N=N, seed=seed, train_time_s=tt, t1_override=ovr)
            r.update(R.seed_row(df)); rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N-train", default="20,60"); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=500); ap.add_argument("--tasks-per-ep", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=2000); ap.add_argument("--arrival-rate", type=float, default=6.0)
    ap.add_argument("--eps-decay-steps", type=int, default=20000)
    ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--out", default="results_safety")
    ap.add_argument("--stub-policy", default="", help="round_robin|random: test without torch")
    args = ap.parse_args()
    a = dict(episodes=args.episodes, tasks_per_ep=args.tasks_per_ep, n_test=args.n_test, arrival_rate=args.arrival_rate,
             eps_decay_steps=args.eps_decay_steps, ctrl=dict(R.CTRL_PRESETS["none"]), stub=args.stub_policy)
    os.makedirs(args.out, exist_ok=True); f = f"{args.out}/per_seed.csv"
    done = set()
    if os.path.exists(f):
        d = pd.read_csv(f); done = set(zip(d.N_train, d.seed))
    jobs = [(int(n), s) for n in args.N_train.split(",") for s in range(args.seed_start, args.seed_start + args.seeds) if (int(n), s) not in done]
    print(f"{len(jobs)} jobs ({len(done)} done)", flush=True); t0 = time.time()
    def save(rows): pd.DataFrame(rows).to_csv(f, mode="a", header=not os.path.exists(f), index=False)
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            futs = {ex.submit(run_job, *j, a): j for j in jobs}
            for i, fu in enumerate(as_completed(futs), 1):
                save(fu.result()); print(f"[{i}/{len(jobs)}] {futs[fu]} {time.time()-t0:.0f}s", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            save(run_job(*j, a)); print(f"[{i}/{len(jobs)}] {j} {time.time()-t0:.0f}s", flush=True)
    ps = pd.read_csv(f)
    print(ps[ps.test_env.isin(["default", "combined"])].groupby(["N_train", "test_env", "variant"])[["viol_t1", "energy_mean", "t1_override"]].mean().round(4))

if __name__ == "__main__":
    main()
