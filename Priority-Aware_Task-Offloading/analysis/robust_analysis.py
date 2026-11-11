"""robust_analysis.py -- analysis of the robustness (R2.6) and extra-baseline (R2.5) runs.
Run from the package root after run_robust.py:  python analysis/robust_analysis.py
Inputs : raw/robust_*/per_seed.csv
Outputs: results/robust_all_per_seed.csv, robust_summary.csv, robust_paired_vs_ref.csv, robust_shift_vs_default.csv,
         figures/Fig_robust_sizes.png, Fig_robust_workloads.png, Fig_baselines_default.png
Statistics as in the main analysis: t-based 95% CI (df = n-1), exact Wilcoxon signed-rank on paired seeds,
Holm correction within each (N_train, test_env, metric) family over the comparison methods (paired_vs_ref) and
within each (arm, N_train, metric) over the test environments (shift_vs_default)."""
import glob, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REF = "dueling_ddqn"
METR = ["energy_mean", "time_t1_mean", "viol_t1", "viol_overall", "eval_reward"]
ps = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("raw/robust_*/per_seed.csv"))], ignore_index=True)
ps = ps.drop_duplicates(["arm", "N_train", "test_env", "seed"], keep="first")
ps.to_csv("results/robust_all_per_seed.csv", index=False)

def tci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    return x.mean(), (stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan)
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj
def wil(d): return 1.0 if np.allclose(d, 0) else stats.wilcoxon(d).pvalue

rows = []
for (arm, nt, env), g in ps.groupby(["arm", "N_train", "test_env"]):
    r = dict(arm=arm, N_train=nt, test_env=env, test_N=int(g.test_N.iloc[0]), n=len(g))
    for m in METR: r[m], r[m + "_ci95"] = tci(g[m])
    rows.append(r)
summ = pd.DataFrame(rows); summ.to_csv("results/robust_summary.csv", index=False)

rows = []
for (nt, env), g in ps.groupby(["N_train", "test_env"]):
    b = g[g.arm == REF].set_index("seed")
    if b.empty: continue
    for m in METR:
        blk = []
        for arm, h in g[g.arm != REF].groupby("arm"):
            h = h.set_index("seed"); c = b.index.intersection(h.index)
            if len(c) < 5: continue
            d = (h.loc[c, m] - b.loc[c, m]).values
            blk.append(dict(N_train=nt, test_env=env, metric=m, arm=arm, ref=REF, n_pairs=len(c), mean_diff=d.mean(), p=wil(d)))
        if blk:
            for x, q in zip(blk, holm([x["p"] for x in blk])): x["holm_p"] = q
            rows += blk
pd.DataFrame(rows).to_csv("results/robust_paired_vs_ref.csv", index=False)

rows = []
for (arm, nt), g in ps.groupby(["arm", "N_train"]):
    b = g[g.test_env == "default"].set_index("seed")
    for m in METR:
        blk = []
        for env, h in g[g.test_env != "default"].groupby("test_env"):
            h = h.set_index("seed"); c = b.index.intersection(h.index)
            if len(c) < 5: continue
            d = (h.loc[c, m] - b.loc[c, m]).values; mean, ci = tci(d)
            blk.append(dict(arm=arm, N_train=nt, metric=m, test_env=env, n_pairs=len(c), mean_diff=mean, ci95=ci, p=wil(d)))
        if blk:
            for x, q in zip(blk, holm([x["p"] for x in blk])): x["holm_p"] = q
            rows += blk
pd.DataFrame(rows).to_csv("results/robust_shift_vs_default.csv", index=False)

ORDER = ["default", "N=20", "N=40", "N=60", "N=80", "N=100", "burst", "tier_shift", "heavy_tasks", "poor_channel", "cap_drop", "combined"]
SHOW = ["dueling_ddqn", "ppo", "oracle_onestep", "heuristic_pa", "greedy", "round_robin"]
def bars(envs, fn, metric="viol_t1", ylabel="Tier-1 deadline violations (%)", scale=1.0):
    nts = sorted(summ.N_train.unique()); nts = [n for n in nts if n in (20, 60)] or nts
    fg, axs = plt.subplots(len(nts), 1, figsize=(10, 3.4 * len(nts)), squeeze=False)
    for ax, nt in zip(axs[:, 0], nts):
        arms = [a for a in SHOW if a in summ[summ.N_train == nt].arm.unique()]
        w = 0.8 / max(1, len(arms)); present = [e for e in envs if e in summ[summ.N_train == nt].test_env.unique()]
        for k, arm in enumerate(arms):
            s = summ[(summ.N_train == nt) & (summ.arm == arm)].set_index("test_env").reindex(present)
            ax.bar(np.arange(len(present)) + k * w, s[metric] * scale, w, yerr=s[metric + "_ci95"] * scale, capsize=1.5, label=arm)
        ax.set_xticks(np.arange(len(present)) + 0.4 - w / 2); ax.set_xticklabels(present, fontsize=8)
        ax.set_title(f"agents trained at N = {nt}", fontsize=9); ax.set_ylabel(ylabel, fontsize=8); ax.grid(axis="y", alpha=.3)
    h, l = axs[0, 0].get_legend_handles_labels(); fg.legend(h, l, loc="lower center", ncol=6, fontsize=7, frameon=False)
    fg.tight_layout(rect=(0, 0.05, 1, 1)); fg.savefig(fn, dpi=200); plt.close(fg)
bars(["default", "N=20", "N=40", "N=60", "N=80", "N=100"], "figures/Fig_robust_sizes.png")
bars(["default", "burst", "tier_shift", "heavy_tasks", "poor_channel", "cap_drop", "combined"], "figures/Fig_robust_workloads.png")
d = summ[summ.test_env == "default"]
fg, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for k, (m, yl, sc) in enumerate([("viol_t1", "Tier-1 violations (%)", 1.0), ("energy_mean", "Energy per task (mJ)", 1e3)]):
    for j, arm in enumerate([a for a in SHOW if a in d.arm.unique()]):
        s = d[d.arm == arm].sort_values("N_train")
        ax[k].errorbar(s.N_train + (j - 2) * 0.8, s[m] * sc, yerr=s[m + "_ci95"] * sc, marker="o", ms=4, capsize=2, label=arm)
    ax[k].set_xticks([20, 40, 60]); ax[k].set_xlabel("N (trained and tested at N)"); ax[k].set_ylabel(yl); ax[k].grid(alpha=.3)
h, l = ax[0].get_legend_handles_labels(); fg.legend(h, l, loc="lower center", ncol=6, fontsize=7, frameon=False)
fg.tight_layout(rect=(0, 0.1, 1, 1)); fg.savefig("figures/Fig_baselines_default.png", dpi=200); plt.close(fg)
pd.set_option("display.width", 200)
print(summ[summ.test_env == "default"][["arm", "N_train", "energy_mean", "viol_t1", "viol_overall"]].round(4).to_string(index=False))
