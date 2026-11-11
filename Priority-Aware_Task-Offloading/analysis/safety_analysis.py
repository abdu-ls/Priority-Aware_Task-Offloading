"""safety_analysis.py -- analysis of the Tier-1 shield experiment (run_safety.py).
Run from the package root:  python analysis/safety_analysis.py
Inputs : raw/safety/per_seed.csv
Outputs: results/safety_summary.csv (mean, t-based 95% CI, worst seed per variant x N_train x test_env),
         results/safety_paired_vs_unprotected.csv (exact Wilcoxon on paired seeds, Holm over the 3 protected variants within
         each N_train x test_env x metric), figures/Fig_safety_seeds.png (per-seed Tier-1 violation, unprotected vs shield)."""
import glob, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
METR = ["viol_t1", "time_t1_mean", "t1_p99_ms", "energy_mean", "viol_overall", "t1_override"]
ps = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("raw/safety*/per_seed.csv"))], ignore_index=True)
ps = ps.drop_duplicates(["variant", "N_train", "test_env", "seed"], keep="first")
def tci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    return x.mean(), (stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan)
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj
def wil(d): return 1.0 if np.allclose(d, 0) else stats.wilcoxon(d).pvalue
rows = []
for (v, nt, env), g in ps.groupby(["variant", "N_train", "test_env"]):
    r = dict(variant=v, N_train=nt, test_env=env, n=len(g), worst_seed_viol_t1=g.viol_t1.max())
    for m in METR: r[m], r[m + "_ci95"] = tci(g[m])
    rows.append(r)
pd.DataFrame(rows).to_csv("results/safety_summary.csv", index=False)
rows = []
for (nt, env), g in ps.groupby(["N_train", "test_env"]):
    b = g[g.variant == "unprotected"].set_index("seed")
    for m in METR[:-1]:
        blk = []
        for v in ["shield_0.8", "shield_0.5", "t1_rule"]:
            x = g[g.variant == v].set_index("seed"); idx = b.index.intersection(x.index)
            d = (x.loc[idx, m] - b.loc[idx, m]).values
            blk.append(dict(N_train=nt, test_env=env, metric=m, variant=v, n_pairs=len(idx), mean_diff=d.mean(), p=wil(d)))
        adj = holm([r["p"] for r in blk])
        for r, a in zip(blk, adj): r["holm_p"] = a; rows.append(r)
pd.DataFrame(rows).to_csv("results/safety_paired_vs_unprotected.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
for a, (nt, env) in zip(ax, [(60, "default"), (60, "combined")]):
    g = ps[(ps.N_train == nt) & (ps.test_env == env)]
    for v, c in [("unprotected", "#b03a48"), ("shield_0.8", "#1b6a7a"), ("shield_0.5", "#c8782a")]:
        x = g[g.variant == v].sort_values("seed"); a.plot(x.seed, x.viol_t1, "o-", ms=3, lw=1, color=c, label=v)
    a.set_title(f"trained at N={nt}, {env}", fontsize=8); a.set_xlabel("seed"); a.grid(alpha=.3)
ax[0].set_ylabel("Tier-1 violations (%)"); ax[0].legend(fontsize=7, frameon=False); fig.tight_layout()
fig.savefig("figures/Fig_safety_seeds.png", dpi=200)
t = pd.read_csv("results/safety_summary.csv")
print(t[t.test_env.isin(["default", "combined", "cap_drop"])].pivot_table(index=["N_train", "test_env", "variant"], values=["viol_t1", "worst_seed_viol_t1", "energy_mean", "t1_override"]).round(4))
