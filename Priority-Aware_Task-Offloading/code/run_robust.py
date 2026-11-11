"""run_robust.py -- robustness and extra-baseline experiments added for the revision.

  R2.6  zero-shot generalisation: a trained agent (no re-training) is tested in
        - other network sizes N = 20, 40, 60, 80, 100 (three MEC servers in all cases, so N >= 80 is a stress test),
        - shifted workloads at its training size: bursty arrivals, a different tier mix (15/25/60 %), heavier tasks
          (10/30/60 % light/medium/heavy), poorer channels (SNR 0-20 dB), a 50 % drop of every MEC server's CPU after
          1,000 of the 2,000 test decisions, and all of these together ("combined").
  R2.5  extra baselines under the same environment, state and reward:
        ppo           -- Proximal Policy Optimisation agent (ppo_agent.py), same training budget as the DQN agents
        oracle_onestep-- non-learning reference that sees the true outcome (delay, energy) of each action for the
                         current task and picks the one with the best reward; no look-ahead beyond the current task.
Everything is trained in the DEFAULT environment (ideal controller). Output rows: one per arm x N_train x test_env x seed.
"""
import argparse, itertools, os, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from iomt_env import IoMTEnv, compute_reward, ACTION_SIZE
import run_v2 as R

SIZES = [20, 40, 60, 80, 100]
_BURST, _TIER, _SNR, _CAP = (1.6, 0.4, 1.0), (0.15, 0.25, 0.60), (0.0, 20.0), (1000, 0.5)
WORKLOADS = {
    "burst":        dict(burst=_BURST),
    "tier_shift":   dict(tier_probs=_TIER),
    "heavy_tasks":  dict(type_probs=(0.1, 0.3, 0.6)),
    "poor_channel": dict(snr_range=_SNR),
    "cap_drop":     dict(cap_drop=_CAP),
    "combined":     dict(burst=_BURST, tier_probs=_TIER, snr_range=_SNR, cap_drop=_CAP),
}
ARMS = ["dueling_ddqn", "ppo", "oracle_onestep", "greedy", "heuristic_pa", "round_robin", "random", "always_local"]
PLAN = {"dueling_ddqn": [20, 60], "ppo": [20, 40, 60]}           # others: 20, 40, 60 (no training)
LEARNED = ("dueling_ddqn", "ppo")


def test_envs(n_train):
    envs = [("default", n_train, {})]
    envs += [(f"N={k}", k, {}) for k in SIZES if k != n_train]
    envs += [(name, n_train, kw) for name, kw in WORKLOADS.items()]
    return envs


def make_policy(arm, seed):
    if arm == "oracle_onestep":
        def act(s, env):
            best, bv = 0, -np.inf
            for k in range(ACTION_SIZE):
                v = compute_reward(env.peek(k, queries=False))
                if v > bv: best, bv = k, v
            return best
        return act, False
    cfg = R.ARMS[arm]; pol = cfg["make"](seed)
    return (lambda s, env: pol.act(s)), cfg.get("queries", True)


def train_ppo(N, seed, a):
    import torch; torch.set_num_threads(1)
    from ppo_agent import PPOAgent
    env = IoMTEnv(N, seed=seed, arrival_rate=a["arrival_rate"]); s = env.reset()
    agent = PPOAgent(len(s), ACTION_SIZE, seed=seed); conv, t0 = [], time.perf_counter()
    for ep in range(a["episodes"]):
        tot = 0.0
        for _ in range(a["tasks_per_ep"]):
            act = agent.act(s); s2, info = env.step(act, queries=True)
            r = compute_reward(info); agent.record(r); tot += r; s = s2
            agent.maybe_update(s)
        conv.append(dict(episode=ep + 1, avg_reward=tot / a["tasks_per_ep"], epsilon=float("nan")))
    return agent, conv, time.perf_counter() - t0


def evaluate_env(act_fn, N, seed, a, queries, kw):
    env = IoMTEnv(N, seed=R.EVAL_OFFSET + seed, arrival_rate=a["arrival_rate"], **kw); s = env.reset()
    rows, inf = [], []
    for _ in range(a["n_test"]):
        t0 = time.perf_counter(); act = act_fn(s, env); inf.append(time.perf_counter() - t0)
        s, info = env.step(act, queries=queries)
        rows.append(dict(priority=info["priority"], energy=info["energy"], time_ms=info["time"] * 1e3,
                         violated=info["violated"], action=info["action"], reward=compute_reward(info)))
    return pd.DataFrame(rows), 1e3 * float(np.mean(inf))


def run_job(arm, n_train, seed, a):
    np.random.seed(seed); conv, tt = [], np.nan
    if arm == "dueling_ddqn":
        agent, conv, tt = R.train_drl(R.ARMS[arm], n_train, seed, a); keep = R.ARMS[arm]["keep"]
        act_fn, queries = (lambda s, env: agent.act(s[keep], greedy=True)), True
    elif arm == "ppo":
        agent, conv, tt = train_ppo(n_train, seed, a)
        act_fn, queries = (lambda s, env: agent.act(s, greedy=True)), True
    else:
        act_fn, queries = make_policy(arm, seed)
    rows = []
    for name, N, kw in test_envs(n_train):
        df, inf_ms = evaluate_env(act_fn, N, seed, a, queries, kw)
        r = dict(arm=arm, N_train=n_train, test_env=name, test_N=N, seed=seed, train_time_s=tt, infer_ms=inf_ms)
        r.update(R.seed_row(df)); rows.append(r)
    for c in conv: c.update(arm=arm, N=n_train, seed=seed)
    return rows, conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS)); ap.add_argument("--N-train", default=None)
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=500); ap.add_argument("--tasks-per-ep", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=2000); ap.add_argument("--arrival-rate", type=float, default=6.0)
    ap.add_argument("--eps-decay-steps", type=int, default=20000)
    ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--out", default="results_robust")
    args = ap.parse_args()
    a = dict(episodes=args.episodes, tasks_per_ep=args.tasks_per_ep, n_test=args.n_test, arrival_rate=args.arrival_rate,
             eps_decay_steps=args.eps_decay_steps, ctrl=dict(R.CTRL_PRESETS["none"]))
    pairs = []
    for arm in args.arms.split(","):
        assert arm in ARMS, f"unknown arm {arm}; choices {ARMS}"
        Ns = [int(x) for x in args.N_train.split(",")] if args.N_train else PLAN.get(arm, [20, 40, 60])
        pairs += [(arm, n) for n in Ns]
    os.makedirs(args.out, exist_ok=True)
    f_seed, f_conv = f"{args.out}/per_seed.csv", f"{args.out}/convergence.csv"
    done = set()
    if os.path.exists(f_seed):
        d = pd.read_csv(f_seed); done = set(zip(d.arm, d.N_train, d.seed))
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    jobs = [(arm, n, s) for (arm, n) in pairs for s in seeds if (arm, n, s) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already done)", flush=True)

    def save(rows, conv):
        pd.DataFrame(rows).to_csv(f_seed, mode="a", header=not os.path.exists(f_seed), index=False)
        if conv: pd.DataFrame(conv).to_csv(f_conv, mode="a", header=not os.path.exists(f_conv), index=False)

    t0 = time.time()
    if args.jobs > 1:
        with ProcessPoolExecutor(args.jobs) as ex:
            futs = {ex.submit(run_job, *j, a): j for j in jobs}
            for i, f in enumerate(as_completed(futs), 1):
                save(*f.result()); print(f"[{i}/{len(jobs)}] {futs[f]}  {time.time()-t0:.0f}s", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            save(*run_job(*j, a)); print(f"[{i}/{len(jobs)}] {j}  {time.time()-t0:.0f}s", flush=True)
    ps = pd.read_csv(f_seed)
    print(ps[ps.test_env == "default"].groupby(["N_train", "arm"])[["energy_mean", "viol_t1", "viol_overall"]].mean().round(4))

if __name__ == "__main__":
    main()
