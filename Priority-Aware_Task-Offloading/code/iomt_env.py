"""
iomt_env.py  --  Repaired IoMT task-offloading environment (v2)

What changed relative to simulator.py (v1), and why
---------------------------------------------------
1. Priority thresholds now follow the paper's Table 4 (v1 classified normal SpO2 and
   most normal blood pressure as "urgent", which produced the 61.5 % Tier-2 mix).
   Gaps in Table 4 are closed conservatively (assigned to Warning). Hypotension is
   added as Critical (<70 mmHg systolic). Tier-1 deadline is 50 ms for every vital.
   Readings are sampled FROM a tier band, so the tier mix is a stated input.
2. Compute demand and payload are drawn from a task TYPE (light / medium / heavy)
   that is INDEPENDENT of the tier. In v1 the CPU-cycle ranges were disjoint per tier,
   so the tier could be inferred from C_i, and local execution always met every
   deadline (always-local was optimal). Now heavy tasks cannot meet 50/100 ms locally.
3. Real contention. MEC servers have a FIFO work backlog that drains at the server
   speed; offloading adds work, so (a) actions change future states and (b) N matters
   through the aggregate arrival rate. Each device also has a local queue.
4. Transitions are real: next_state is the state of the next arrival (v1 used a
   fictitious next task for the same device while the next training step used a
   random device).
5. The queue feature is meaningful and identical in training and evaluation.
6. SimPy is not used. This is an event-driven simulation with an explicit clock
   (Poisson arrivals, queue draining between arrivals). Describe it that way in the paper.
7. Common random numbers: every exogenous draw is independent of the agent's action
   and consumed in a fixed order, so two agents evaluated with the same env seed see
   identical task / channel / WAN streams.
"""
import numpy as np

# ---- physical constants --------------------------------------------------------
BANDWIDTH_HZ = 10e6
KAPPA        = 1e-28            # effective switched capacitance
CLOUD_CPU_HZ = 50e9
BATTERY_J    = 0.2 * 3600 * 3.7 # 200 mAh @ 3.7 V = 2664 J

# ---- normalisation constants (state features in ~[0,1]) -----------------------
C_MAX  = 2e8                    # max CPU cycles of a task
D_MAX  = 100e3 * 8              # max payload, bits (100 KB)
Q_REF  = 0.5                    # s of local backlog mapped to 1.0
H_LOAD = 0.1                    # s of full-speed MEC work mapped to load 1.0
R_MAX  = 100e6                  # bit/s

DEADLINE = {1: 0.050, 2: 0.100, 3: 5.0}

# state layout (13-D, same as the paper's Eq. (1))
IDX = dict(B=0, f=1, Q=2, D=3, C=4, P=5, R=6, L=slice(7, 10), SNR=slice(10, 13))
STATE_SIZE, ACTION_SIZE = 13, 5

# ---- physiological thresholds (Table 4, gaps closed conservatively) ------------
# v < c_lo -> critical | c_lo <= v < w_lo -> warning | w_lo <= v <= n_hi -> normal
# n_hi < v <= w_hi -> warning | v > w_hi -> critical
THRESH = {
    "heart_rate":     dict(c_lo=45, w_lo=60, n_hi=100, w_hi=150, lo=25, hi=200),
    "spo2":           dict(c_lo=90, w_lo=95, n_hi=100, w_hi=100, lo=70, hi=100),
    "blood_glucose":  dict(c_lo=54, w_lo=70, n_hi=140, w_hi=250, lo=30, hi=400),
    "blood_pressure": dict(c_lo=70, w_lo=90, n_hi=120, w_hi=180, lo=50, hi=240),  # systolic
}
VITALS = list(THRESH)

def classify_priority(vital, v):
    t = THRESH[vital]
    if v < t["c_lo"] or v > t["w_hi"]:
        return 1
    if v < t["w_lo"] or v > t["n_hi"]:
        return 2
    return 3

def _sample_value(vital, tier, rng):
    t = THRESH[vital]
    if tier == 3:
        return rng.uniform(t["w_lo"], t["n_hi"])
    if tier == 2:
        bands = [(t["c_lo"], t["w_lo"])]
        if t["w_hi"] > t["n_hi"]:
            bands.append((t["n_hi"], t["w_hi"]))
    else:
        bands = [(t["lo"], t["c_lo"])]
        if t["hi"] > t["w_hi"]:
            bands.append((t["w_hi"], t["hi"]))
    w = np.array([b - a for a, b in bands]); k = rng.choice(len(bands), p=w / w.sum())
    lo, hi = bands[k]
    return rng.uniform(lo + 1e-6, hi - 1e-6)

# task types: (cycles range, payload-bits range); independent of tier
TASK_TYPES = [
    ("light",  (1e6, 1e7),  (1e3 * 8,  10e3 * 8)),
    ("medium", (1e7, 5e7),  (10e3 * 8, 50e3 * 8)),
    ("heavy",  (5e7, 2e8),  (20e3 * 8, 100e3 * 8)),
]


class IoMTEnv:
    def __init__(self, n_devices=20, n_mec=3, seed=0, arrival_rate=6.0,
                 tier_probs=(0.05, 0.15, 0.80), type_probs=(0.3, 0.4, 0.3),
                 ctrl_query_s=0.0, flow_s=0.0, ctrl_msg_bits=0.0,
                 burst=None, cap_drop=None, snr_range=(5.0, 30.0)):
        """arrival_rate: tasks per second PER DEVICE (aggregate = n_devices * arrival_rate)."""
        self.n, self.m = n_devices, n_mec
        # control-plane cost (all zero = ideal controller, identical to the results of the first runs):
        #  ctrl_query_s  : delay of one state query to the SDN controller (s), paid before the decision
        #                  by every method that uses controller-supplied state (queries=True in step()).
        #  ctrl_msg_bits : size of the query message (bits); its device energy is ptx * bits / mean uplink rate.
        #  flow_s        : reactive flow-rule installation delay (s), paid by every offloaded task (MEC or cloud)
        #                  by every method, whether or not it queries the controller.
        self.ctrl_query_s, self.flow_s, self.ctrl_msg_bits = ctrl_query_s, flow_s, ctrl_msg_bits
        # --- non-stationarity options for the robustness tests (all off by default = original simulator) ---
        #  burst=(hi, lo, dwell_s): two-state arrival process; the arrival rate is multiplied by hi or lo and the
        #        state flips after exponentially distributed dwell times (own RNG stream, so the task/channel
        #        streams of the main generator are not disturbed). hi=1.6, lo=0.4 keeps the mean rate unchanged.
        #  cap_drop=(after_decisions, factor): every MEC server's CPU is multiplied by `factor` after that many decisions.
        #  snr_range=(lo, hi): range (dB) of the uniform per-link SNR draw (default 5-30 dB).
        self.burst, self.cap_drop, self.snr_range = burst, cap_drop, tuple(snr_range)
        self._seed = seed
        self.lam = arrival_rate * n_devices
        self.tier_probs, self.type_probs = np.array(tier_probs), np.array(type_probs)
        self.rng = np.random.default_rng(seed)
        # hardware (fixed for the lifetime of the env)
        self.cpu     = self.rng.uniform(0.5e9, 1.0e9, n_devices)
        self.ptx     = self.rng.uniform(0.1, 0.5, n_devices)
        self.batt0   = BATTERY_J * self.rng.uniform(0.6, 1.0, n_devices)
        self.mec_cpu0 = self.rng.uniform(5e9, 10e9, n_mec)
        self.mec_cpu = self.mec_cpu0.copy()
        self.reset()

    # -- dynamics ------------------------------------------------------------------
    def reset(self):
        self.t = 0.0
        self.n_steps = 0
        self.mec_cpu = self.mec_cpu0.copy()
        if self.burst is not None:                     # burst state uses its own RNG stream
            self._rng_b = np.random.default_rng(self._seed + 99991)
            self._bstate = 0; self._bnext = self._rng_b.exponential(self.burst[2])
        self.busy_until = np.zeros(self.n)             # local queues
        self.batt = self.batt0.copy()
        self.backlog = np.zeros(self.m)                # MEC work backlog (cycles, as of self.t)
        self.bg = self.rng.uniform(0.1, 0.4, self.m)   # other tenants' share of each server
        self._arrival()
        return self.state

    def _advance(self, dt):
        eff = self.mec_cpu * (1.0 - self.bg)
        self.backlog = np.maximum(0.0, self.backlog - eff * dt)
        noise = self.rng.normal(size=self.m)           # always drawn
        k = min(dt / 0.2, 1.0)
        self.bg = np.clip(self.bg + 0.3 * k * (0.25 - self.bg) + 0.03 * np.sqrt(min(dt, 0.5) / 0.1) * noise, 0.05, 0.5)
        self.t += dt

    def _arrival(self):
        rng = self.rng
        lam = self.lam
        if self.burst is not None:
            while self.t >= self._bnext:
                self._bstate ^= 1; self._bnext += self._rng_b.exponential(self.burst[2])
            lam = self.lam * self.burst[self._bstate]
        self._advance(rng.exponential(1.0 / lam))
        d = int(rng.integers(self.n))
        vital = VITALS[int(rng.integers(len(VITALS)))]
        tier = 1 + int(rng.choice(3, p=self.tier_probs))
        value = _sample_value(vital, tier, rng)
        prio = classify_priority(vital, value)
        assert prio == tier
        ty = int(rng.choice(len(TASK_TYPES), p=self.type_probs))
        _, (c0, c1), (d0, d1) = TASK_TYPES[ty]
        self.task = dict(device=d, priority=prio, deadline=DEADLINE[prio], vital=vital, value=value,
                         ttype=TASK_TYPES[ty][0], cycles=rng.uniform(c0, c1), bits=rng.uniform(d0, d1))
        self.snr_db = rng.uniform(self.snr_range[0], self.snr_range[1], self.m)   # per-link SNR for THIS decision
        self.wan = rng.uniform(0.020, 0.080)           # drawn every step (CRN)
        self.rates = BANDWIDTH_HZ * np.log2(1.0 + 10 ** (self.snr_db / 10))
        self.state = self._build_state()

    def _build_state(self):
        tk, d = self.task, self.task["device"]
        load = np.clip(self.bg + self.backlog / (self.mec_cpu * H_LOAD), 0.0, 1.0)
        return np.concatenate([[
            self.batt[d] / BATTERY_J,
            self.cpu[d] / 1e9,
            min(max(0.0, self.busy_until[d] - self.t) / Q_REF, 1.0),
            tk["bits"] / D_MAX,
            tk["cycles"] / C_MAX,
            tk["priority"] / 3.0,
            min(self.rates.mean() / R_MAX, 1.0)],
            load, self.snr_db / 30.0]).astype(np.float32)

    def _outcome(self, action, queries=False):
        """Time, energy and the state update of `action` for the current task. Does not modify the environment."""
        tk, d = self.task, self.task["device"]
        C, D, f = tk["cycles"], tk["bits"], self.cpu[d]
        lead = self.ctrl_query_s if queries else 0.0          # time before the task can start
        e_sig = (self.ptx[d] * self.ctrl_msg_bits / self.rates.mean()) if queries else 0.0
        upd = None
        if action == 0:                                       # local (FIFO device queue)
            start = max(self.t + lead, self.busy_until[d])
            finish = start + C / f
            upd = ("busy", d, finish)
            time, energy = finish - self.t, KAPPA * f * f * C
        elif 1 <= action <= self.m:                           # MEC j
            j = action - 1
            rate, eff = self.rates[j], self.mec_cpu[j] * (1.0 - self.bg[j])
            tx = D / rate
            arrive = lead + self.flow_s + tx                  # time until the data are at the server
            wait = max(0.0, self.backlog[j] - eff * arrive) / eff
            time, energy = arrive + wait + C / eff, self.ptx[d] * tx
            upd = ("backlog", j, max(self.backlog[j], eff * arrive) + C)
        else:                                                 # cloud via best access link
            tx = D / self.rates.max()
            time, energy = lead + self.flow_s + tx + self.wan + C / CLOUD_CPU_HZ, self.ptx[d] * tx
        return time, energy + e_sig, upd

    def peek(self, action, queries=False):
        """Read-only look-ahead used by the one-step reference controller: info dict of `action`."""
        time, energy, _ = self._outcome(action, queries)
        tk = self.task
        return dict(time=time, energy=energy, violated=int(time > tk["deadline"]), action=int(action),
                    priority=tk["priority"], deadline=tk["deadline"])

    def step(self, action, queries=False):
        """queries=True: the decision used controller-supplied state, so the task pays the query delay
        and the device pays the query-message energy. Offloaded tasks always pay flow_s."""
        tk, d = self.task, self.task["device"]
        time, energy, upd = self._outcome(action, queries)
        if upd is not None:
            if upd[0] == "busy": self.busy_until[upd[1]] = upd[2]
            else: self.backlog[upd[1]] = upd[2]
        self.batt[d] = max(0.0, self.batt[d] - energy)
        info = dict(time=time, energy=energy, violated=int(time > tk["deadline"]), action=int(action),
                    priority=tk["priority"], deadline=tk["deadline"], ttype=tk["ttype"], device=d)
        self.n_steps += 1
        if self.cap_drop is not None and self.n_steps == self.cap_drop[0]:
            self.mec_cpu = self.mec_cpu0 * self.cap_drop[1]
        self._arrival()
        return self.state, info


# ---- centralised reward (all scripts must call this) -----------------------------
def compute_reward(info, alpha=0.6, beta=0.4, lam=50.0, e_ref=0.02,
                   uniform_weight=False, deadline_override=None):
    """-(alpha * w * min(T/Tmax,3) + beta * min(E/e_ref,1) + lam * 1[T>Tmax]).
    uniform_weight=True      -> w_i = 1 (no priority weighting)
    deadline_override=value  -> training reward uses one common deadline (tier-blind agent)
    e_ref is 0.02 J here (v1 used 0.5 J, which made the energy term ~1e-3)."""
    dl = deadline_override if deadline_override else info["deadline"]
    w = 1.0 if uniform_weight else 4.0 - info["priority"]
    return -(alpha * w * min(info["time"] / dl, 3.0)
             + beta * min(info["energy"] / e_ref, 1.0)
             + lam * float(info["time"] > dl))
