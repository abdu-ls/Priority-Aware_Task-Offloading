"""ctrlplane_analysis.py -- effect of control-plane cost (controller query delay, signalling energy, flow-rule installation).
Run from the package root after the control-plane runs:  python analysis/ctrlplane_analysis.py
Inputs : results/all_per_seed.csv (scenario 'none' = ideal controller, the original runs)
         raw/ctrl_<none|low|mid|high>_*/per_seed.csv (every folder whose name starts with ctrl_)
Outputs: results/ctrlplane_all_per_seed.csv, results/ctrlplane_summary.csv, results/ctrlplane_paired_vs_ref.csv,
         results/ctrlplane_delta_vs_none.csv, results/ctrlplane_device_local_vs_ref.csv, figures/Fig_ctrlplane_tier1.png, figures/Fig_ctrlplane_energy.png
Scenarios: none (0/0 ms), low (query 2 ms, flow 5 ms), mid (5/10 ms), high (10/20 ms); 500-byte query message."""
import glob, os, re, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SC = ["none", "low", "mid", "high"]; REF = "dueling_ddqn"
METR = ["energy_mean", "time_t1_mean", "viol_t1", "viol_overall", "eval_reward"]
base = pd.read_csv("results/all_per_seed.csv"); base = base[base.N.isin([20, 60])].assign(scenario="none")
parts = [base]
for f in sorted(glob.glob("raw/ctrl_*/per_seed.csv")):
    sc = re.match(r"ctrl_(none|low|mid|high)", os.path.basename(os.path.dirname(f))).group(1)
    parts.append(pd.read_csv(f).assign(scenario=sc))
ps = pd.concat(parts, ignore_index=True)
ps = ps.drop_duplicates(["scenario", "arm", "N", "seed"], keep="first")
ps["scenario"] = pd.Categorical(ps.scenario, SC, ordered=True)
ps.to_csv("results/ctrlplane_all_per_seed.csv", index=False)

def tci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    return x.mean(), (stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan)
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj
def wil(d): return 1.0 if np.allclose(d, 0) else stats.wilcoxon(d).pvalue

# 1. summary
rows = []
for (sc, N, arm), g in ps.groupby(["scenario", "N", "arm"], observed=True):
    r = dict(scenario=sc, N=N, arm=arm, n=len(g))
    for m in METR: r[m], r[m + "_ci95"] = tci(g[m])
    rows.append(r)
summ = pd.DataFrame(rows).sort_values(["N", "arm", "scenario"]); summ.to_csv("results/ctrlplane_summary.csv", index=False)

# 2. paired vs reference within each scenario and N (Holm over the comparison arms)
rows = []
for (sc, N), g in ps.groupby(["scenario", "N"], observed=True):
    b = g[g.arm == REF].set_index("seed")
    if b.empty: continue
    for m in METR:
        blk = []
        for arm, h in g[g.arm != REF].groupby("arm"):
            h = h.set_index("seed"); c = b.index.intersection(h.index)
            if len(c) < 5: continue
            d = (h.loc[c, m] - b.loc[c, m]).values
            blk.append(dict(scenario=sc, N=N, metric=m, arm=arm, ref=REF, n_pairs=len(c), mean_diff=d.mean(), p=wil(d)))
        if blk:
            for x, q in zip(blk, holm([x["p"] for x in blk])): x["holm_p"] = q
            rows += blk
pd.DataFrame(rows).to_csv("results/ctrlplane_paired_vs_ref.csv", index=False)

# 3. change of each arm relative to the ideal controller (paired by seed), Holm over the three scenarios
rows = []
for (N, arm), g in ps.groupby(["N", "arm"], observed=True):
    b = g[g.scenario == "none"].set_index("seed")
    for m in METR:
        blk = []
        for sc in SC[1:]:
            h = g[g.scenario == sc].set_index("seed"); c = b.index.intersection(h.index)
            if len(c) < 5: continue
            d = (h.loc[c, m] - b.loc[c, m]).values
            mean, ci = tci(d); blk.append(dict(N=N, arm=arm, metric=m, scenario=sc, n_pairs=len(c), mean_diff=mean, ci95=ci, p=wil(d)))
        if blk:
            for x, q in zip(blk, holm([x["p"] for x in blk])): x["holm_p"] = q
            rows += blk
pd.DataFrame(rows).to_csv("results/ctrlplane_delta_vs_none.csv", index=False)

# 3b. value of controller information: device_local (no query) minus dueling_ddqn (queries), paired by seed.
#     Holm over the four scenarios, separately for each N and metric.
rows = []
for N, gN in ps.groupby("N"):
    for m in METR:
        blk = []
        for sc in SC:
            g = gN[gN.scenario == sc]
            b = g[g.arm == REF].set_index("seed"); h = g[g.arm == "device_local"].set_index("seed"); c = b.index.intersection(h.index)
            if len(c) < 5: continue
            d = (h.loc[c, m] - b.loc[c, m]).values; mean, ci = tci(d)
            blk.append(dict(N=N, metric=m, scenario=sc, n_pairs=len(c), mean_diff=mean, ci95=ci, p=wil(d)))
        if blk:
            for x, q in zip(blk, holm([x["p"] for x in blk])): x["holm_p"] = q
            rows += blk
pd.DataFrame(rows).to_csv("results/ctrlplane_device_local_vs_ref.csv", index=False)

# 4. figures
def fig(metric, ylabel, fn, scale=1.0):
    arms = [a for a in summ.arm.unique()]
    fg, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    for k, N in enumerate([20, 60]):
        for arm in arms:
            s = summ[(summ.N == N) & (summ.arm == arm)].sort_values("scenario")
            if len(s) < 2: continue
            x = [SC.index(v) for v in s.scenario]
            ax[k].errorbar(x, s[metric] * scale, yerr=s[metric + "_ci95"] * scale, marker="o", ms=4, lw=1.2, capsize=2, label=arm)
        ax[k].set_xticks(range(4)); ax[k].set_xticklabels(["none\n0/0", "low\n2/5", "mid\n5/10", "high\n10/20"])
        ax[k].set_xlabel("control-plane cost: query / flow-install (ms)"); ax[k].set_title(f"N = {N}"); ax[k].grid(alpha=.3)
    ax[0].set_ylabel(ylabel); h, l = ax[1].get_legend_handles_labels(); fg.legend(h, l, loc="lower center", ncol=5, fontsize=7, frameon=False); fg.tight_layout(rect=(0, 0.1, 1, 1)); fg.savefig(fn, dpi=200); plt.close(fg)
os.makedirs("figures", exist_ok=True)
fig("viol_t1", "Tier-1 deadline violations (%)", "figures/Fig_ctrlplane_tier1.png")
fig("energy_mean", "Energy per task (mJ)", "figures/Fig_ctrlplane_energy.png", 1e3)
print(summ[["scenario", "N", "arm", "energy_mean", "time_t1_mean", "viol_t1", "viol_overall"]].round(4).to_string(index=False))
