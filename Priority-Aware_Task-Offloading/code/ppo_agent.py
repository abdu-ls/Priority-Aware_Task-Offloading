"""ppo_agent.py -- Proximal Policy Optimisation baseline (PyTorch), added for the revision (reviewer comment R2.5).

Same state (13-D) and the same reward as the DQN agents. Actor and critic are separate MLPs with the same 2 x 128 ReLU
trunk as the DQN. The task stream is a continuing process (no terminal states), so GAE bootstraps from V(s') at the end
of every rollout. Hyper-parameters are common defaults and were NOT tuned for this problem (same for the DQN agents).
  rollout 128 decisions, 8 epochs, minibatch 32, lr 3e-4, gamma 0.95, GAE lambda 0.95, clip 0.2,
  entropy coefficient 0.01, value coefficient 0.5, gradient clipping 1.0, rewards scaled by 0.1 for the update.
Training budget is the same as for the DQN agents: 500 episodes x 100 decisions = 50,000 decisions.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def gae(rewards, values, last_value, gamma, lam):
    """Generalised advantage estimation for a continuing task. rewards, values: 1-D arrays of length T."""
    T = len(rewards); adv = np.zeros(T, dtype=np.float32); nxt_v, nxt_a = last_value, 0.0
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * nxt_v - values[t]
        nxt_a = delta + gamma * lam * nxt_a
        adv[t] = nxt_a; nxt_v = values[t]
    return adv, adv + np.asarray(values, dtype=np.float32)


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n_out))


class PPOAgent:
    def __init__(self, state_size, action_size, seed=0, lr=3e-4, gamma=0.95, lam=0.95, clip=0.2, epochs=8,
                 minibatch=32, rollout=128, ent=0.01, vf=0.5, max_grad=1.0, reward_scale=0.1):
        torch.manual_seed(seed); self.rng = np.random.default_rng(seed)
        self.pi, self.v = mlp(state_size, action_size), mlp(state_size, 1)
        self.opt = optim.Adam(list(self.pi.parameters()) + list(self.v.parameters()), lr=lr)
        self.gamma, self.lam, self.clip, self.epochs, self.mb = gamma, lam, clip, epochs, minibatch
        self.rollout, self.ent, self.vf, self.max_grad, self.rs = rollout, ent, vf, max_grad, reward_scale
        self.S, self.A, self.LP, self.V, self.R = [], [], [], [], []

    @torch.no_grad()
    def act(self, s, greedy=False):
        x = torch.as_tensor(np.asarray(s, dtype=np.float32)).unsqueeze(0)
        logits = self.pi(x)[0]
        if greedy:
            return int(torch.argmax(logits))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        self.S.append(np.asarray(s, dtype=np.float32)); self.A.append(int(a))
        self.LP.append(float(dist.log_prob(a))); self.V.append(float(self.v(x)[0, 0]))
        return int(a)

    def record(self, reward):
        """Call once after every sampled action (stores the reward of the transition)."""
        self.R.append(float(reward) * self.rs)

    def maybe_update(self, s_next):
        if len(self.R) < self.rollout:
            return None
        with torch.no_grad():
            last_v = float(self.v(torch.as_tensor(np.asarray(s_next, dtype=np.float32)).unsqueeze(0))[0, 0])
        adv, ret = gae(np.asarray(self.R, dtype=np.float32), np.asarray(self.V, dtype=np.float32), last_v,
                       self.gamma, self.lam)
        S = torch.as_tensor(np.stack(self.S)); A = torch.as_tensor(np.asarray(self.A, dtype=np.int64))
        LP0 = torch.as_tensor(np.asarray(self.LP, dtype=np.float32))
        ADV = torch.as_tensor(adv); RET = torch.as_tensor(ret)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n, last = len(self.R), 0.0
        for _ in range(self.epochs):
            perm = self.rng.permutation(n)
            for i in range(0, n, self.mb):
                idx = torch.as_tensor(perm[i:i + self.mb])
                dist = torch.distributions.Categorical(logits=self.pi(S[idx]))
                lp = dist.log_prob(A[idx]); ratio = torch.exp(lp - LP0[idx])
                surr = torch.min(ratio * ADV[idx], torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * ADV[idx])
                vloss = ((self.v(S[idx]).squeeze(-1) - RET[idx]) ** 2).mean()
                loss = -surr.mean() + self.vf * vloss - self.ent * dist.entropy().mean()
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(self.pi.parameters()) + list(self.v.parameters()), self.max_grad)
                self.opt.step(); last = loss.item()
        self.S, self.A, self.LP, self.V, self.R = [], [], [], [], []
        return last
