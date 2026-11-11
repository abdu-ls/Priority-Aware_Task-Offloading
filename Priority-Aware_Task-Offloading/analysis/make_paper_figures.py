# Figures 2-4 of the paper (learning curves, control-plane cost, robustness). Run from the package root after the other analysis scripts.
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.makedirs('figures/paper',exist_ok=True)
R='results/'
O='figures/paper/'
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
col={'dueling_ddqn':'#1b6a7a','device_local':'#c8782a','greedy':'#b03a48','heuristic_pa':'#6b5b95','ppo':'#8a8a8a'}
lab={'dueling_ddqn':'D3QN (with controller state)','device_local':'D3QN, device-local state','greedy':'Greedy','heuristic_pa':'Priority-aware rule','ppo':'PPO'}
c=pd.read_csv(R+'ctrlplane_summary.csv').drop_duplicates(['scenario','N','arm'])
sc=['none','low','mid','high']; x=np.arange(4)
fig,ax=plt.subplots(1,2,figsize=(7,2.7),sharey=False)
for a,N in zip(ax,[20,60]):
    for arm in ['dueling_ddqn','device_local','greedy','heuristic_pa']:
        d=c[(c.N==N)&(c.arm==arm)].set_index('scenario').loc[sc]
        a.errorbar(x,d.viol_t1,d.viol_t1_ci95,color=col[arm],marker='o',ms=3,lw=1.2,capsize=2,label=lab[arm])
    a.set_xticks(x);a.set_xticklabels(['none\n0/0','low\n2/5','mid\n5/10','high\n10/20'])
    a.set_title(f'N = {N}');a.set_xlabel('query / flow-installation delay (ms)')
    a.grid(axis='y',alpha=.3)
ax[0].set_ylabel('Tier-1 violations (%)')
h,l=ax[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,frameon=False,fontsize=7)
fig.tight_layout(rect=(0,.1,1,1));fig.savefig(O+'ctrl.png',dpi=300)
r=pd.read_csv(R+'robust_summary.csv')
envs=['default','N=40','N=60','N=80','N=100','burst','tier_shift','heavy_tasks','poor_channel','cap_drop','combined']
names=['default','N=40','N=60','N=80','N=100','burst','tier\nshift','heavy\ntasks','poor\nchannel','cap.\ndrop','combined']
fig,ax=plt.subplots(2,1,figsize=(7,5.2))
for a,nt in zip(ax,[20,60]):
    envs_nt=[e if e!='default' else 'default' for e in envs]
    arms=['dueling_ddqn','ppo','heuristic_pa','greedy'];w=.2
    for i,arm in enumerate(arms):
        v=[];e=[]
        for en in envs:
            if en==f'N={nt}': en='default'
            q=r[(r.arm==arm)&(r.N_train==nt)&(r.test_env==en)]
            if q.empty and arm in('greedy','heuristic_pa'): q=r[(r.arm==arm)&(r.test_env==en)].head(1)
            v.append(q.viol_t1.values[0]);e.append(q.viol_t1_ci95.values[0])
        a.bar(np.arange(len(envs))+(i-1.5)*w,v,w,yerr=e,color=col[arm],label=lab[arm],error_kw=dict(lw=.7,capsize=1))
    a.set_xticks(range(len(envs)));a.set_xticklabels(names);a.set_title(f'Agents trained at N = {nt}',fontsize=8);a.set_ylabel('Tier-1 violations (%)');a.grid(axis='y',alpha=.3)
    a.set_ylim(0,100)
h,l=ax[0].get_legend_handles_labels();fig.legend(h,l[:4],ncol=4,frameon=False,fontsize=7,loc='lower center')
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(O+'robust.png',dpi=300)
import shutil;shutil.copy('figures/Fig4.4_training_reward.png',O+'learning.png')
