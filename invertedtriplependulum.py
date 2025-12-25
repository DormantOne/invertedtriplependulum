#!/usr/bin/env python3
"""
================================================================================
🔥 PENDULUM FURNACE v2.2 — CONSTRAINED EVOLUTION + DIVERSITY INJECTION
    Triple Pendulum Control with Smarter Genome Constraints
================================================================================

Changes from v2.1:
  ✓ CONSTRAINED GENOME RANGES:
    - Spectral radius capped at 1.5 (was 2.2) — prevent instability
    - Minimum reservoir size 400 (was 200) — more capacity
    - Minimum density 0.08 (was 0.03) — ensure connectivity
    - Alive weight has MINIMUM 0.15 after softmax — must care about survival
  
  ✓ DIVERSITY INJECTION:
    - Every 50 generations, inject 2 random individuals
    - Helps escape local optima
    - Triggered also if no improvement for 30 generations
  
  ✓ COMPOUND FITNESS:
    - fitness = mean_steps * (1 + 0.1 * normalized_return)
    - Values both survival AND quality of control
  
  ✓ STAGNATION DETECTION:
    - Track generations without improvement
    - Boost mutation (sigma *= 1.5) when stagnant
    - Reset boost when improvement found
  
  ✓ REWARD WEIGHT STRUCTURE:
    - Alive weight guaranteed minimum via clamping
    - Prevents degenerate "maximize instant tip height" strategies

================================================================================
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed
from flask import Flask, jsonify, Response

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = torch.device("cpu")

CORE_COUNT = max(1, min(4, os.cpu_count() or 1))
POP_MULT = 4
POP_SIZE = CORE_COUNT * POP_MULT

# Genome: 9 liquid hyperparams + 5 reward weights = 14
HYPER_D = 9
REWARD_D = 5
GENOME_DIM = HYPER_D + REWARD_D  # 14

# Inner training / eval budgets
TRAIN_EPISODES_INITIAL = 40
TRAIN_EPISODES_VARSITY = 150

# Varsity threshold
VARSITY_STEP_THRESHOLD = 140.0

# Evaluation
EVAL_EPISODES = 4
SCOUT_EPISODES = 2
MAX_STEPS = 900

# Integration
DT_PHYS = 0.002
CONTROL_DT = 0.02
SUBSTEPS = int(round(CONTROL_DT / DT_PHYS))
SUBSTEPS = max(1, SUBSTEPS)

# Motor limits
TAU_MAX = 24.0

# Actor-Critic hyperparams
GAMMA = 0.99
GRAD_CLIP = 1.0
VALUE_COEF = 0.50
ENTROPY_COEF = 0.003

# Richer observations: 12D proprio + 3D feedback = 15D
OBS_DIM = 12
ACTION_DIM = 3
INPUT_DIM = OBS_DIM + ACTION_DIM

# Intrinsic plasticity
IP_LEARNING_RATE = 0.001
IP_TARGET_MEAN = 0.0
IP_TARGET_VAR = 0.2

# Reward
REWARD_SCALE = 2.2
TERMINAL_FALL_PENALTY = -10.0

# Persistence
SAVE_FILE = "pendulum_furnace_best_v22.json"
WEIGHTS_FILE = "pendulum_furnace_best_v22.pt"
SIGMA_FILE = "pendulum_furnace_sigma_v22.json"
BEST_METRIC_TAG = "steps_v5"

# Demo
DEMO_FPS = 50
AUTO_DEMO_STEPS = 420.0

# === NATURAL GRADIENT EVOLUTION PARAMS ===
INIT_SIGMA = 0.06
SIGMA_MIN = 0.004
SIGMA_MAX = 0.20

SENSITIVITY_SMOOTHING = 0.85
SENSITIVITY_FLOOR = 0.01

SIGMA_GROW_RATE = 1.08
SIGMA_SHRINK_RATE = 0.92

SENSITIVITY_THRESHOLD = 0.5

N_ELITES = 5

# Crossover
CROSSOVER_PROB = 0.3
BLX_ALPHA = 0.3

# === NEW v2.2: Diversity and stagnation ===
DIVERSITY_INJECT_EVERY = 50       # Inject randoms every N generations
DIVERSITY_INJECT_COUNT = 2        # How many randoms to inject
STAGNATION_THRESHOLD = 30         # Generations without improvement
STAGNATION_SIGMA_BOOST = 1.5      # Multiply sigma by this when stagnant

# === NEW v2.2: Genome constraints ===
MIN_RESERVOIR_SIZE = 400          # Was 200
MAX_RESERVOIR_SIZE = 1400         # Was 1400
MIN_DENSITY = 0.08                # Was 0.03
MAX_SPECTRAL_RADIUS = 1.5         # Was 2.2
MIN_ALIVE_WEIGHT = 0.15           # Guaranteed minimum after softmax

# Demo control
DEMO_LOG_EVERY_N_RESETS = 5
DEMO_EXIT_AFTER_RESETS = 80
DEMO_EXIT_AFTER_CONSEC_SHORT = 12
DEMO_SHORT_FALL_STEPS = 40

# Viz
VIZ_MAX_NEURONS = 200
VIZ_MAX_LINKS = 400

# Gene names for logging
GENE_NAMES = ["SIZE", "DENS", "LK_F", "LK_M", "LK_S", 
              "PL_F", "PL_S", "RAD", "LR",
              "W_ALV", "W_TIP", "W_UP", "W_VEL", "W_TAU"]

# =============================================================================
# SHARED STATE
# =============================================================================

SYSTEM_STATE: Dict[str, Any] = {
    "status": "INITIALIZING…",
    "generation": 0,
    "best_score": -1e9,
    "best_genome": None,
    "best_weights": None,
    "mode": "TRAINING",
    "logs": [],
    "hyperparams": {},
    "current_id": "Waiting…",
    "manual_demo_request": False,
    "demo_stop": False,
    "demo_resets": 0,
    "pop_vectors": [],
    "history_vectors": [],
    "sim_view": {},
    "brain_view": {},
    "viz_mode": "hyperspace",
    "gene_sigma": None,
    "gene_sensitivity": None,
    "stagnation_count": 0,
}

_STATE_LOCK = threading.Lock()


def add_log(msg: str) -> None:
    with _STATE_LOCK:
        SYSTEM_STATE["logs"].insert(0, msg)
        if len(SYSTEM_STATE["logs"]) > 25:
            SYSTEM_STATE["logs"].pop()
    print(f"[PENDULUM_FURNACE] {msg}")


# =============================================================================
# GENETICS — v2.2 with constraints
# =============================================================================

def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


class GeneDecoder:
    """
    14D genome -> hyperparameters + reward weights
    
    v2.2 CONSTRAINTS:
    - Reservoir size: 400-1400 (was 200-1400)
    - Density: 0.08-0.40 (was 0.03-0.40)
    - Spectral radius: 0.6-1.5 (was 0.6-2.2)
    - Alive weight: minimum 0.15 after softmax
    """
    @staticmethod
    def decode(vector: np.ndarray) -> Dict[str, Any]:
        v = np.clip(np.asarray(vector, dtype=np.float64), 0.0, 1.0)

        # === CONSTRAINED: larger minimum reservoir ===
        n_res = int(v[0] * (MAX_RESERVOIR_SIZE - MIN_RESERVOIR_SIZE) + MIN_RESERVOIR_SIZE)
        
        # === CONSTRAINED: higher minimum density ===
        density = v[1] * (0.40 - MIN_DENSITY) + MIN_DENSITY
        
        leak_fast = v[2] * 0.45 + 0.40
        leak_med = v[3] * 0.33 + 0.12
        leak_slow = v[4] * 0.13 + 0.02
        
        pool_fast = v[5] * 0.35 + 0.15
        pool_slow = v[6] * 0.35 + 0.15
        if pool_fast + pool_slow > 0.85:
            scale = 0.85 / (pool_fast + pool_slow)
            pool_fast *= scale
            pool_slow *= scale
        pool_med = 1.0 - pool_fast - pool_slow
        
        # === CONSTRAINED: lower max spectral radius ===
        radius = v[7] * (MAX_SPECTRAL_RADIUS - 0.6) + 0.6
        
        lr = 10 ** (-4.2 + (2.0 * v[8]))

        input_gain = 1.8

        # Compute base softmax weights
        raw = (v[9:14] * 4.0) - 2.0
        w = _softmax_np(raw).astype(np.float64)
        
        # === CONSTRAINED: ensure minimum alive weight ===
        # If alive weight is below minimum, redistribute from others
        if w[0] < MIN_ALIVE_WEIGHT:
            deficit = MIN_ALIVE_WEIGHT - w[0]
            w[0] = MIN_ALIVE_WEIGHT
            # Take proportionally from others
            other_sum = w[1:].sum()
            if other_sum > 1e-6:
                w[1:] = w[1:] * (1.0 - MIN_ALIVE_WEIGHT) / other_sum
            else:
                w[1:] = (1.0 - MIN_ALIVE_WEIGHT) / 4.0
        
        # Renormalize to ensure sum = 1
        w = w / w.sum()

        return {
            "n_reservoir": n_res,
            "density": float(density),
            "leak_fast": float(leak_fast),
            "leak_med": float(leak_med),
            "leak_slow": float(leak_slow),
            "pool_fast": float(pool_fast),
            "pool_med": float(pool_med),
            "pool_slow": float(pool_slow),
            "spectral_radius": float(radius),
            "lr": float(lr),
            "input_gain": float(input_gain),
            "reward_w": [float(x) for x in w.tolist()],
        }


# =============================================================================
# NATURAL GRADIENT EVOLUTION + DIVERSITY
# =============================================================================

class NaturalGradientEvolution:
    """
    Adaptive per-gene mutation with crossover and diversity injection.
    
    v2.2 additions:
    - Stagnation detection and sigma boosting
    - Periodic diversity injection
    """
    
    def __init__(self, genome_dim: int = GENOME_DIM):
        self.dim = genome_dim
        self.sigma = np.ones(genome_dim) * INIT_SIGMA
        self.sensitivity = np.ones(genome_dim) * 0.5
        self.sigma_history = []
        self.stagnation_count = 0
        self.last_best = -1e9
        
    def mutate(self, parent: np.ndarray) -> np.ndarray:
        """Mutate parent genome using adaptive per-gene sigma."""
        noise = np.random.randn(self.dim) * self.sigma
        child = np.clip(parent + noise, 0.0, 1.0)
        return child
    
    def crossover_blx(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """BLX-α blend crossover."""
        child = np.zeros(self.dim)
        for i in range(self.dim):
            p1, p2 = parent1[i], parent2[i]
            lo, hi = min(p1, p2), max(p1, p2)
            d = hi - lo
            alpha_d = BLX_ALPHA * d
            child[i] = np.random.uniform(lo - alpha_d, hi + alpha_d)
        return np.clip(child, 0.0, 1.0)
    
    def check_stagnation(self, current_best: float) -> bool:
        """Check if we're stagnating and boost sigma if so."""
        if current_best > self.last_best + 1.0:  # Meaningful improvement
            self.stagnation_count = 0
            self.last_best = current_best
            return False
        else:
            self.stagnation_count += 1
            if self.stagnation_count >= STAGNATION_THRESHOLD:
                # Boost sigma to escape local optimum
                self.sigma = np.clip(self.sigma * STAGNATION_SIGMA_BOOST, SIGMA_MIN, SIGMA_MAX)
                add_log(f"⚡ Stagnation detected ({self.stagnation_count} gens) — boosting σ")
                self.stagnation_count = 0  # Reset counter
                return True
            return False
    
    def reproduce(self, elites: List[Dict[str, Any]], pop_size: int, 
                  generation: int) -> List[Dict[str, Any]]:
        """
        Create new population from elites with diversity injection.
        """
        new_pop = []
        
        # Keep elites unchanged
        for e in elites:
            new_pop.append({
                "genome": np.array(e["genome"]),
                "weights": e.get("weights")
            })
        
        # Periodic diversity injection
        inject_diversity = (generation % DIVERSITY_INJECT_EVERY == 0) or (self.stagnation_count > STAGNATION_THRESHOLD // 2)
        n_random = DIVERSITY_INJECT_COUNT if inject_diversity else 0
        
        if n_random > 0:
            add_log(f"🌱 Injecting {n_random} random individuals for diversity")
        
        # Add random individuals for diversity
        for _ in range(n_random):
            if len(new_pop) < pop_size:
                new_pop.append({
                    "genome": np.random.rand(self.dim),
                    "weights": None  # Fresh start
                })
        
        # Fill rest with offspring
        while len(new_pop) < pop_size:
            if random.random() < CROSSOVER_PROB and len(elites) >= 2:
                p1, p2 = random.sample(elites, 2)
                child_genome = self.crossover_blx(
                    np.array(p1["genome"]), 
                    np.array(p2["genome"])
                )
                child_genome = child_genome + np.random.randn(self.dim) * self.sigma * 0.5
                child_genome = np.clip(child_genome, 0.0, 1.0)
                weights = p1.get("weights") if p1.get("mean_steps", 0) >= p2.get("mean_steps", 0) else p2.get("weights")
                new_pop.append({"genome": child_genome, "weights": weights})
            else:
                parent = random.choice(elites)
                child_genome = self.mutate(np.array(parent["genome"]))
                new_pop.append({"genome": child_genome, "weights": parent.get("weights")})
        
        return new_pop
    
    def update_from_generation(self, results: List[Dict[str, Any]]) -> None:
        """Update sensitivity estimates and sigma based on generation results."""
        if len(results) < 4:
            return
        
        results_sorted = sorted(results, key=lambda r: r["mean_steps"], reverse=True)
        n_elite = max(2, len(results) // 4)
        elites = results_sorted[:n_elite]
        others = results_sorted[n_elite:]
        
        if len(others) < 2:
            return
        
        elite_genomes = np.array([r["genome"] for r in elites])
        other_genomes = np.array([r["genome"] for r in others])
        
        elite_mean = elite_genomes.mean(axis=0)
        other_mean = other_genomes.mean(axis=0)
        gene_diff = np.abs(elite_mean - other_mean)
        
        all_genomes = np.array([r["genome"] for r in results])
        gene_variance = all_genomes.var(axis=0) + 1e-6
        
        raw_sensitivity = gene_diff / (np.sqrt(gene_variance) + 1e-6)
        raw_sensitivity = raw_sensitivity / (raw_sensitivity.max() + 1e-6)
        
        self.sensitivity = (SENSITIVITY_SMOOTHING * self.sensitivity + 
                          (1 - SENSITIVITY_SMOOTHING) * raw_sensitivity)
        self.sensitivity = np.clip(self.sensitivity, SENSITIVITY_FLOOR, 1.0)
        
        for i in range(self.dim):
            if self.sensitivity[i] > SENSITIVITY_THRESHOLD:
                self.sigma[i] *= SIGMA_SHRINK_RATE
            else:
                self.sigma[i] *= SIGMA_GROW_RATE
        
        self.sigma = np.clip(self.sigma, SIGMA_MIN, SIGMA_MAX)
        
        self.sigma_history.append(self.sigma.copy())
        if len(self.sigma_history) > 100:
            self.sigma_history.pop(0)
    
    def get_sigma_summary(self) -> str:
        sens_order = np.argsort(self.sensitivity)[::-1]
        most_sens = sens_order[:3]
        least_sens = sens_order[-3:]
        
        parts = ["σ: "]
        for i in most_sens:
            parts.append(f"{GENE_NAMES[i]}={self.sigma[i]:.3f}↓")
        parts.append(" | ")
        for i in least_sens:
            parts.append(f"{GENE_NAMES[i]}={self.sigma[i]:.3f}↑")
        
        return " ".join(parts)
    
    def save(self, filepath: str) -> None:
        data = {
            "sigma": self.sigma.tolist(),
            "sensitivity": self.sensitivity.tolist(),
            "stagnation_count": self.stagnation_count,
            "last_best": self.last_best,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)
    
    def load(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            self.sigma = np.array(data["sigma"])
            self.sensitivity = np.array(data["sensitivity"])
            self.stagnation_count = data.get("stagnation_count", 0)
            self.last_best = data.get("last_best", -1e9)
            return True
        except:
            return False


# =============================================================================
# PHYSICS
# =============================================================================

@dataclass
class Params:
    l1: float = 0.85
    l2: float = 0.85
    l3: float = 0.85
    m1: float = 2.2
    m2: float = 1.7
    m3: float = 1.2
    lc1: Optional[float] = None
    lc2: Optional[float] = None
    lc3: Optional[float] = None
    Izz1: Optional[float] = None
    Izz2: Optional[float] = None
    Izz3: Optional[float] = None
    g: float = 9.81
    viscous_b1: float = 0.40
    viscous_b2: float = 0.28
    viscous_b3: float = 0.20
    coulomb_c1: float = 0.18
    coulomb_c2: float = 0.13
    coulomb_c3: float = 0.10
    coulomb_smooth_k: float = 25.0


def _fill_defaults(p: Params) -> Params:
    if p.lc1 is None: p.lc1 = p.l1 / 2.0
    if p.lc2 is None: p.lc2 = p.l2 / 2.0
    if p.lc3 is None: p.lc3 = p.l3 / 2.0
    if p.Izz1 is None: p.Izz1 = p.m1 * (p.l1 ** 2) / 12.0
    if p.Izz2 is None: p.Izz2 = p.m2 * (p.l2 ** 2) / 12.0
    if p.Izz3 is None: p.Izz3 = p.m3 * (p.l3 ** 2) / 12.0
    return p


def _skew(a: np.ndarray) -> np.ndarray:
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    return np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]], dtype=np.float64)


def _rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _Xtrans(r: np.ndarray) -> np.ndarray:
    r = r.astype(np.float64)
    I = np.eye(3, dtype=np.float64)
    Z = np.zeros((3, 3), dtype=np.float64)
    return np.block([[I, Z], [-_skew(r), I]])


def _crm(v: np.ndarray) -> np.ndarray:
    w, u = v[0:3], v[3:6]
    Z = np.zeros((3, 3), dtype=np.float64)
    return np.block([[_skew(w), Z], [_skew(u), _skew(w)]])


def _crf(v: np.ndarray) -> np.ndarray:
    return -_crm(v).T


def _spatial_inertia(m: float, com: np.ndarray, Izz: float) -> np.ndarray:
    Ic = np.diag([Izz, 1e-9, Izz]).astype(np.float64)
    c = com.astype(np.float64)
    C = _skew(c)
    I0 = Ic + m * (C.T @ C)
    return np.vstack([np.hstack([I0, m * C.T]), np.hstack([m * C, m * np.eye(3)])])


class TriplePendulumModel:
    def __init__(self, p: Params):
        self.p = _fill_defaults(p)
        self.S = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.Xtree = [
            np.eye(6, dtype=np.float64),
            _Xtrans(np.array([0.0, self.p.l1, 0.0])),
            _Xtrans(np.array([0.0, self.p.l2, 0.0])),
        ]
        self.I = [
            _spatial_inertia(self.p.m1, np.array([0.0, self.p.lc1, 0.0]), self.p.Izz1),
            _spatial_inertia(self.p.m2, np.array([0.0, self.p.lc2, 0.0]), self.p.Izz2),
            _spatial_inertia(self.p.m3, np.array([0.0, self.p.lc3, 0.0]), self.p.Izz3),
        ]
        self.a0 = np.array([0.0, 0.0, 0.0, 0.0, -self.p.g, 0.0], dtype=np.float64)

    def friction_tau(self, qd: np.ndarray) -> np.ndarray:
        p = self.p
        def fric(v, b, c):
            return -b * v - c * math.tanh(p.coulomb_smooth_k * v)
        return np.array([fric(qd[0], p.viscous_b1, p.coulomb_c1),
                         fric(qd[1], p.viscous_b2, p.coulomb_c2),
                         fric(qd[2], p.viscous_b3, p.coulomb_c3)], dtype=np.float64)

    def forward_dynamics(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray) -> np.ndarray:
        S, Xtree, Ibody = self.S, self.Xtree, self.I
        n = 3
        Xup = [None] * n
        v = [np.zeros(6, dtype=np.float64) for _ in range(n)]
        c = [np.zeros(6, dtype=np.float64) for _ in range(n)]
        IA = [Ibody[i].copy() for i in range(n)]
        pA = [np.zeros(6, dtype=np.float64) for _ in range(n)]
        U = [np.zeros(6, dtype=np.float64) for _ in range(n)]
        d = np.zeros(n, dtype=np.float64)
        u = np.zeros(n, dtype=np.float64)

        for i in range(n):
            XJ = np.block([[_rotz(q[i]), np.zeros((3,3))], [np.zeros((3,3)), _rotz(q[i])]])
            Xup[i] = XJ @ Xtree[i]
            vJ = S * qd[i]
            v[i] = vJ if i == 0 else Xup[i] @ v[i-1] + vJ
            c[i] = _crm(v[i]) @ vJ
            pA[i] = _crf(v[i]) @ (IA[i] @ v[i])

        for i in range(n-1, -1, -1):
            U[i] = IA[i] @ S
            d[i] = S @ U[i]
            u[i] = tau[i] - S @ pA[i]
            if i > 0:
                Ia = IA[i] - np.outer(U[i], U[i]) / d[i]
                pa = pA[i] + Ia @ c[i] + U[i] * (u[i] / d[i])
                IA[i-1] += Xup[i].T @ Ia @ Xup[i]
                pA[i-1] += Xup[i].T @ pa

        a = [np.zeros(6, dtype=np.float64) for _ in range(n)]
        qdd = np.zeros(n, dtype=np.float64)
        for i in range(n):
            a[i] = (Xup[i] @ self.a0 + c[i]) if i == 0 else (Xup[i] @ a[i-1] + c[i])
            qdd[i] = (u[i] - U[i] @ a[i]) / d[i]
            a[i] += S * qdd[i]
        return qdd


def fk_points(q: np.ndarray, p: Params) -> Tuple[Tuple[float, float], ...]:
    th1, th2, th3 = q[0], q[0]+q[1], q[0]+q[1]+q[2]
    x1, y1 = p.l1*math.sin(th1), p.l1*math.cos(th1)
    x2, y2 = x1 + p.l2*math.sin(th2), y1 + p.l2*math.cos(th2)
    x3, y3 = x2 + p.l3*math.sin(th3), y2 + p.l3*math.cos(th3)
    return ((0.0, 0.0), (x1, y1), (x2, y2), (x3, y3))


def compute_com(q: np.ndarray, p: Params) -> Tuple[float, float]:
    th1, th2, th3 = q[0], q[0]+q[1], q[0]+q[1]+q[2]
    x1_com = p.lc1 * math.sin(th1)
    y1_com = p.lc1 * math.cos(th1)
    x2_com = p.l1*math.sin(th1) + p.lc2*math.sin(th2)
    y2_com = p.l1*math.cos(th1) + p.lc2*math.cos(th2)
    x3_com = p.l1*math.sin(th1) + p.l2*math.sin(th2) + p.lc3*math.sin(th3)
    y3_com = p.l1*math.cos(th1) + p.l2*math.cos(th2) + p.lc3*math.cos(th3)
    total_mass = p.m1 + p.m2 + p.m3
    com_x = (p.m1*x1_com + p.m2*x2_com + p.m3*x3_com) / total_mass
    com_y = (p.m1*y1_com + p.m2*y2_com + p.m3*y3_com) / total_mass
    return (com_x, com_y)


# =============================================================================
# ENVIRONMENT
# =============================================================================

class PendulumEnv:
    def __init__(self, model: TriplePendulumModel, p: Params, reward_w: np.ndarray):
        self.model = model
        self.p = p
        self.reach = p.l1 + p.l2 + p.l3
        self.reward_w = np.asarray(reward_w, dtype=np.float64).reshape(REWARD_D,)
        self.prev_com = None
        self.reset()

    def reset(self) -> Tuple[torch.Tensor, np.ndarray]:
        q = np.array([math.radians(np.random.uniform(-8, 8)),
                      math.radians(np.random.uniform(-5, 5)),
                      math.radians(np.random.uniform(-3, 3))], dtype=np.float64)
        qd = np.zeros(3, dtype=np.float64)
        self.x = np.concatenate([q, qd])
        self.t = 0.0
        self.steps = 0
        self.prev_tau = np.zeros(3, dtype=np.float64)
        self.prev_com = compute_com(q, self.p)
        return self.get_obs(), self.prev_tau

    def _ground_contact(self, q: np.ndarray) -> bool:
        pts = fk_points(q, self.p)
        for i in range(1, 4):
            y = pts[i][1]
            if not np.isfinite(y) or y <= 0.0:
                return True
        return False

    def _done(self, q: np.ndarray, qd: np.ndarray) -> bool:
        if self._ground_contact(q):
            return True
        if np.any(~np.isfinite(qd)):
            return True
        return False

    def _reward(self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray) -> float:
        pts = fk_points(q, self.p)
        tip_y = pts[-1][1]
        tip_norm = float(np.clip(tip_y / self.reach, -0.5, 1.2)) / 1.2
        
        joint_heights = [pts[i][1] for i in range(1, 4)]
        min_height = min(joint_heights)
        min_height_norm = float(np.clip(min_height / self.reach, -0.5, 1.0))
        
        th = np.array([q[0], q[0]+q[1], q[0]+q[1]+q[2]])
        err = float(np.mean(1.0 - np.cos(th)))
        up_norm = float(np.clip(err / 1.5, 0, 1))
        
        vel_norm = float(np.clip(np.sum(qd**2) / 60.0, 0, 1))
        tau_norm = float(np.clip(np.sum((tau/TAU_MAX)**2) / 3.0, 0, 1))
        
        survival_bonus = 1.0 + 0.5 * (self.steps / MAX_STEPS)
        
        c = np.array([
            survival_bonus,
            tip_norm + 0.3*min_height_norm,
            -up_norm,
            -vel_norm,
            -tau_norm
        ])
        
        return float(np.clip(REWARD_SCALE * np.dot(self.reward_w, c), -6, 6))

    def get_obs(self) -> torch.Tensor:
        q, qd = self.x[0:3], self.x[3:6]
        pts = fk_points(q, self.p)
        tip_x, tip_y = pts[-1]
        
        com = compute_com(q, self.p)
        if self.prev_com is not None:
            com_vel = (com[1] - self.prev_com[1]) / CONTROL_DT
        else:
            com_vel = 0.0
        self.prev_com = com
        
        prop = np.array([
            np.sin(q[0]), np.sin(q[1]), np.sin(q[2]),
            np.cos(q[0]), np.cos(q[1]), np.cos(q[2]),
            np.clip(qd[0]/6, -2, 2),
            np.clip(qd[1]/6, -2, 2),
            np.clip(qd[2]/6, -2, 2),
            tip_x / self.reach,
            tip_y / self.reach,
            np.clip(com_vel / 5.0, -2, 2),
        ], dtype=np.float32)
        
        return torch.from_numpy(prop).unsqueeze(0)

    def step(self, tau: np.ndarray) -> Tuple[torch.Tensor, np.ndarray, float, bool]:
        tau = np.clip(np.asarray(tau, dtype=np.float64).reshape(3,), -TAU_MAX, TAU_MAX)
        
        for _ in range(SUBSTEPS):
            q, qd = self.x[0:3], self.x[3:6]
            qdd = self.model.forward_dynamics(q, qd, tau + self.model.friction_tau(qd))
            qd = qd + DT_PHYS * qdd
            q = q + DT_PHYS * qd
            q = (q + math.pi) % (2*math.pi) - math.pi
            self.x[0:3], self.x[3:6] = q, qd
            self.t += DT_PHYS

        self.steps += 1
        self.prev_tau = tau.copy()
        
        q, qd = self.x[0:3], self.x[3:6]
        done = self._done(q, qd) or self.steps >= MAX_STEPS
        r = self._reward(q, qd, tau)
        if done and self.steps < MAX_STEPS:
            r += TERMINAL_FALL_PENALTY
            
        return self.get_obs(), self.prev_tau, r, done


# =============================================================================
# MULTI-TIMESCALE RESERVOIR
# =============================================================================

class MultiTimescaleReservoir(nn.Module):
    def __init__(self, input_dim: int, params: Dict[str, Any]):
        super().__init__()
        
        self.size = int(params["n_reservoir"])
        self.input_dim = input_dim
        
        n_fast = int(self.size * params["pool_fast"])
        n_slow = int(self.size * params["pool_slow"])
        n_med = self.size - n_fast - n_slow
        
        self.pool_sizes = [n_fast, n_med, n_slow]
        self.pool_boundaries = [0, n_fast, n_fast + n_med, self.size]
        
        self.leak_rates = torch.zeros(self.size)
        self.leak_rates[0:n_fast] = params["leak_fast"]
        self.leak_rates[n_fast:n_fast+n_med] = params["leak_med"]
        self.leak_rates[n_fast+n_med:] = params["leak_slow"]
        self.register_buffer("leak", self.leak_rates)
        
        gain = float(params["input_gain"])
        density = float(params["density"])
        radius = float(params["spectral_radius"])

        self.w_in = nn.Linear(input_dim, self.size, bias=False)
        with torch.no_grad():
            in_mask = (torch.rand(self.size, input_dim) < 0.6).float()
            in_weights = (torch.rand(self.size, input_dim) * 2.0 - 1.0) * gain * in_mask
            self.w_in.weight.copy_(in_weights)
            self.w_in.weight.requires_grad_(False)

        mask = (torch.rand(self.size, self.size) < density).float()
        w_rec = (torch.rand(self.size, self.size) * 2.0 - 1.0) * mask

        try:
            eig = torch.linalg.eigvals(w_rec)
            max_e = torch.max(torch.abs(eig)).item()
        except:
            max_e = torch.linalg.norm(w_rec, ord=2).item()

        if max_e > 1e-6:
            w_rec = w_rec * (radius / max_e)

        self.w_rec = nn.Parameter(w_rec, requires_grad=False)
        self.mask = mask
        self.bias = nn.Parameter(torch.zeros(self.size), requires_grad=False)
        
        self.register_buffer("ip_mean", torch.zeros(self.size))
        self.register_buffer("ip_var", torch.ones(self.size) * 0.1)

        inds = mask.nonzero().tolist()
        stride = max(1, len(inds) // VIZ_MAX_LINKS)
        self.links = inds[::stride][:VIZ_MAX_LINKS]
        self.link_weights = [float(w_rec[i,j].item()) for i,j in self.links]

    def forward(self, u: torch.Tensor, h: torch.Tensor, apply_ip: bool = False) -> torch.Tensor:
        inj = self.w_in(u)
        rec = F.linear(h, self.w_rec)
        pre = inj + rec + self.bias
        act = torch.tanh(pre)
        
        if apply_ip and self.training:
            with torch.no_grad():
                batch_mean = act.mean(dim=0)
                batch_var = act.var(dim=0) + 1e-6
                self.ip_mean = 0.99 * self.ip_mean + 0.01 * batch_mean
                self.ip_var = 0.99 * self.ip_var + 0.01 * batch_var
                mean_err = self.ip_mean - IP_TARGET_MEAN
                self.bias.data -= IP_LEARNING_RATE * mean_err
        
        h_new = (1.0 - self.leak) * h + self.leak * act
        return h_new

    def get_pool_info(self) -> Dict[str, Any]:
        return {
            "pool_sizes": self.pool_sizes,
            "pool_boundaries": self.pool_boundaries,
            "leak_rates": [
                float(self.leak[self.pool_boundaries[0]].item()),
                float(self.leak[self.pool_boundaries[1]].item()) if self.pool_sizes[1] > 0 else 0,
                float(self.leak[self.pool_boundaries[2]].item()) if self.pool_sizes[2] > 0 else 0,
            ]
        }


class LiquidAgent(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()
        self.params = params
        self.reservoir = MultiTimescaleReservoir(INPUT_DIM, params)
        self.actor = nn.Linear(self.reservoir.size, 6)
        self.critic = nn.Linear(self.reservoir.size, 1)

    def init_hidden(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, self.reservoir.size, dtype=torch.float32)

    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor, h: torch.Tensor, 
                apply_ip: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u = torch.cat([obs, prev_action], dim=1)
        h2 = self.reservoir(u, h, apply_ip=apply_ip)
        out = self.actor(h2)
        mu = out[:, 0:3]
        log_std = torch.clamp(out[:, 3:6], -3.0, 1.0)
        v = self.critic(h2).squeeze(1)
        return mu, log_std, v, h2

    def sample(self, obs: torch.Tensor, prev_action: torch.Tensor, h: torch.Tensor,
               apply_ip: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std, v, h2 = self.forward(obs, prev_action, h, apply_ip)
        std = torch.exp(log_std)
        eps = torch.randn_like(mu)
        u = mu + std * eps
        a = torch.tanh(u)
        tau = TAU_MAX * a
        
        normal_logp = -0.5 * (((u - mu) / (std + 1e-8))**2 + 2*log_std + math.log(2*math.pi))
        logp = normal_logp.sum(dim=1) - torch.log(1 - a**2 + 1e-6).sum(dim=1)
        entropy = (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum(dim=1)
        
        return tau, logp, v, entropy, h2

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor, prev_action: torch.Tensor, 
                          h: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor, np.ndarray, float]:
        mu, _, v, h2 = self.forward(obs, prev_action, h)
        tau = TAU_MAX * torch.tanh(mu)
        return tau.squeeze(0).cpu().numpy(), h2, mu.squeeze(0).cpu().numpy(), float(v.item())

    @torch.no_grad()
    def get_activations(self, h: torch.Tensor) -> List[float]:
        h_np = h.squeeze(0).cpu().numpy()
        stride = max(1, len(h_np) // VIZ_MAX_NEURONS)
        return h_np[::stride][:VIZ_MAX_NEURONS].tolist()


# =============================================================================
# WORKER LIFECYCLE — v2.2 with compound fitness
# =============================================================================

def _discounted_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    R = 0.0
    out = []
    for r in reversed(rewards):
        R = r + gamma * R
        out.insert(0, R)
    t = torch.tensor(out, dtype=torch.float32)
    if len(t) > 1:
        t = (t - t.mean()) / (t.std() + 1e-8)
    return t


def run_life_cycle(packet: Dict[str, Any]) -> Dict[str, Any]:
    torch.set_num_threads(1)

    seed = int(packet.get("seed", 0) + 1337)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    genome = np.asarray(packet["genome"], dtype=np.float64)
    gen = int(packet["gen"])
    pretrained = packet.get("weights", None)

    try:
        p = _fill_defaults(Params())
        model = TriplePendulumModel(p)
        decoded = GeneDecoder.decode(genome)
        reward_w = np.asarray(decoded["reward_w"], dtype=np.float64)
        env = PendulumEnv(model, p, reward_w=reward_w)
        params = decoded

        agent = LiquidAgent(params).to(DEVICE)

        titan = False
        if pretrained is not None:
            try:
                agent.load_state_dict(pretrained, strict=True)
                titan = True
            except:
                pass

        lr = float(params["lr"])
        opt = torch.optim.Adam(
            list(agent.actor.parameters()) + list(agent.critic.parameters()),
            lr=lr
        )

        def train_block(episodes: int) -> None:
            agent.train()
            for _ in range(episodes):
                obs, prev_tau = env.reset()
                obs = obs.to(DEVICE)
                prev_action = torch.from_numpy(prev_tau / TAU_MAX).float().unsqueeze(0).to(DEVICE)
                h = agent.init_hidden(1).to(DEVICE)

                logps, vals, ents, rewards = [], [], [], []

                for _ in range(MAX_STEPS):
                    tau_t, logp_t, v_t, ent_t, h = agent.sample(obs, prev_action, h, apply_ip=True)
                    tau_np = tau_t.squeeze(0).detach().cpu().numpy()
                    
                    obs2, new_prev_tau, r, done = env.step(tau_np)
                    obs = obs2.to(DEVICE)
                    prev_action = torch.from_numpy(new_prev_tau / TAU_MAX).float().unsqueeze(0).to(DEVICE)
                    
                    logps.append(logp_t)
                    vals.append(v_t)
                    ents.append(ent_t)
                    rewards.append(r)
                    
                    if done:
                        break

                rets = _discounted_returns(rewards, GAMMA).to(DEVICE)
                logps_t = torch.cat(logps)
                vals_t = torch.cat(vals)
                ents_t = torch.cat(ents)
                adv = rets - vals_t.detach()
                
                loss = -(logps_t * adv).sum() + VALUE_COEF * 0.5 * ((vals_t - rets)**2).sum() - ENTROPY_COEF * ents_t.sum()
                
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP)
                opt.step()

        train_block(TRAIN_EPISODES_INITIAL)

        agent.eval()
        scout_steps = []
        for _ in range(SCOUT_EPISODES):
            obs, prev_tau = env.reset()
            obs = obs.to(DEVICE)
            prev_action = torch.from_numpy(prev_tau / TAU_MAX).float().unsqueeze(0).to(DEVICE)
            h = agent.init_hidden(1).to(DEVICE)
            steps = 0
            for _ in range(MAX_STEPS):
                tau_np, h, _, _ = agent.act_deterministic(obs, prev_action, h)
                obs2, new_prev_tau, _, done = env.step(tau_np)
                obs = obs2.to(DEVICE)
                prev_action = torch.from_numpy(new_prev_tau / TAU_MAX).float().unsqueeze(0).to(DEVICE)
                steps += 1
                if done:
                    break
            scout_steps.append(steps)

        scout_mean_steps = float(np.mean(scout_steps))

        varsity = False
        if scout_mean_steps >= VARSITY_STEP_THRESHOLD:
            varsity = True
            train_block(TRAIN_EPISODES_VARSITY)

        agent.eval()
        eval_returns, eval_steps = [], []
        for _ in range(EVAL_EPISODES):
            obs, prev_tau = env.reset()
            obs = obs.to(DEVICE)
            prev_action = torch.from_numpy(prev_tau / TAU_MAX).float().unsqueeze(0).to(DEVICE)
            h = agent.init_hidden(1).to(DEVICE)
            total, steps = 0.0, 0
            for _ in range(MAX_STEPS):
                tau_np, h, _, _ = agent.act_deterministic(obs, prev_action, h)
                obs2, new_prev_tau, r, done = env.step(tau_np)
                obs = obs2.to(DEVICE)
                prev_action = torch.from_numpy(new_prev_tau / TAU_MAX).float().unsqueeze(0).to(DEVICE)
                total += r
                steps += 1
                if done:
                    break
            eval_returns.append(total)
            eval_steps.append(steps)

        mean_return = float(np.mean(eval_returns))
        mean_steps = float(np.mean(eval_steps))
        
        # === v2.2: COMPOUND FITNESS ===
        # Normalize return to [0, 1] range roughly (typical returns are -50 to +500)
        norm_return = np.clip((mean_return + 50) / 550, 0, 1)
        fitness = mean_steps * (1.0 + 0.1 * norm_return)

        trapped = None
        if mean_steps >= max(0.85 * VARSITY_STEP_THRESHOLD, 100.0):
            trapped = {k: v.detach().cpu() for k, v in agent.state_dict().items()}

        return {
            "fitness": fitness,
            "mean_return": mean_return,
            "mean_steps": mean_steps,
            "scout_mean_steps": scout_mean_steps,
            "genome": genome.tolist(),
            "params": params,
            "id": f"G{gen}-{random.randint(100,999)}",
            "varsity": varsity,
            "titan": titan,
            "weights": trapped,
        }

    except Exception as e:
        return {"fitness": -1e9, "error": str(e), "genome": genome.tolist()}


# =============================================================================
# EVOLUTION ENGINE
# =============================================================================

def load_data() -> None:
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r") as f:
                data = json.load(f)
            g = data.get("genome")
            if g and len(g) == GENOME_DIM:
                with _STATE_LOCK:
                    SYSTEM_STATE["best_genome"] = g
                    SYSTEM_STATE["generation"] = int(data.get("gen", 0))
                    if data.get("metric") == BEST_METRIC_TAG:
                        SYSTEM_STATE["best_score"] = float(data.get("score", -1e9))
                        add_log(f"📂 Loaded save. Best steps: {SYSTEM_STATE['best_score']:.1f}")
                    else:
                        SYSTEM_STATE["best_score"] = -1e9
                        add_log("📂 Loaded genome, metric mismatch — reset score")
        except Exception as e:
            add_log(f"⚠️ Save load error: {e}")

    if os.path.exists(WEIGHTS_FILE):
        try:
            with _STATE_LOCK:
                SYSTEM_STATE["best_weights"] = torch.load(WEIGHTS_FILE, map_location=DEVICE)
            add_log("🧠 Loaded champion weights")
        except Exception as e:
            add_log(f"⚠️ Weight load error: {e}")


def save_data(score: float, gen: int, genome: List[float], weights=None) -> None:
    with open(SAVE_FILE, "w") as f:
        json.dump({"metric": BEST_METRIC_TAG, "score": score, "gen": gen, "genome": genome}, f)
    if weights:
        try:
            torch.save(weights, WEIGHTS_FILE)
        except Exception as e:
            add_log(f"⚠️ Weight save error: {e}")


class EvolutionEngine(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.pop_size = POP_SIZE
        self.population_data = [{"genome": np.random.rand(GENOME_DIM), "weights": None} 
                                for _ in range(self.pop_size)]
        
        self.nge = NaturalGradientEvolution(GENOME_DIM)
        if self.nge.load(SIGMA_FILE):
            add_log("📂 Loaded adaptive sigma state")

    def run_demo_mode(self, agent_data: Dict[str, Any]) -> None:
        with _STATE_LOCK:
            SYSTEM_STATE["mode"] = "DEMO"
            SYSTEM_STATE["demo_stop"] = False
            SYSTEM_STATE["demo_resets"] = 0
            SYSTEM_STATE["current_id"] = agent_data.get("id", "CHAMPION")
            SYSTEM_STATE["status"] = "👁️ WATCHING"

        params = agent_data["params"]
        weights = agent_data.get("weights")
        reward_w = np.asarray(params.get("reward_w", [0.2]*5))

        p = _fill_defaults(Params())
        model = TriplePendulumModel(p)
        env = PendulumEnv(model, p, reward_w=reward_w)

        agent = LiquidAgent(params).to(DEVICE)
        loaded = False

        if weights:
            try:
                agent.load_state_dict(weights, strict=True)
                loaded = True
            except:
                pass

        if not loaded:
            with _STATE_LOCK:
                bw = SYSTEM_STATE.get("best_weights")
            if bw:
                try:
                    agent.load_state_dict(bw, strict=True)
                    loaded = True
                except:
                    pass

        if not loaded:
            add_log("⚠️ Demo has no compatible weights")

        agent.eval()
        obs, prev_tau = env.reset()
        obs = obs.to(DEVICE)
        prev_action = torch.from_numpy(prev_tau / TAU_MAX).float().unsqueeze(0).to(DEVICE)
        h = agent.init_hidden(1).to(DEVICE)

        dt_target = 1.0 / DEMO_FPS
        resets, consec_short, steps_in_ep = 0, 0, 0

        while True:
            t0 = time.time()
            with _STATE_LOCK:
                if SYSTEM_STATE["mode"] != "DEMO" or SYSTEM_STATE["demo_stop"]:
                    break

            tau_np, h, mu_np, v_est = agent.act_deterministic(obs, prev_action, h)
            obs2, new_prev_tau, r, done = env.step(tau_np)
            obs = obs2.to(DEVICE)
            prev_action = torch.from_numpy(new_prev_tau / TAU_MAX).float().unsqueeze(0).to(DEVICE)
            steps_in_ep += 1

            q, qd = env.x[0:3].copy(), env.x[3:6].copy()
            pts = fk_points(q, p)
            activations = agent.get_activations(h)
            pool_info = agent.reservoir.get_pool_info()

            with _STATE_LOCK:
                SYSTEM_STATE["sim_view"] = {
                    "t": env.t, "q": q.tolist(), "qd": qd.tolist(),
                    "tau": tau_np.tolist(), "r": r,
                    "points": [[x, y] for x, y in pts],
                    "tip": list(pts[-1]), "reach": env.reach,
                    "steps": steps_in_ep,
                }
                SYSTEM_STATE["brain_view"] = {
                    "mu": mu_np.tolist(), "v": v_est,
                    "links": agent.reservoir.links,
                    "link_weights": agent.reservoir.link_weights,
                    "activations": activations,
                    "n_reservoir": agent.reservoir.size,
                    "obs": obs.squeeze(0).cpu().numpy().tolist(),
                    "prev_action": (prev_action.squeeze(0).cpu().numpy() * TAU_MAX).tolist(),
                    "pool_info": pool_info,
                }
                SYSTEM_STATE["gene_sigma"] = self.nge.sigma.tolist()
                SYSTEM_STATE["gene_sensitivity"] = self.nge.sensitivity.tolist()
                SYSTEM_STATE["stagnation_count"] = self.nge.stagnation_count

            if done:
                resets += 1
                with _STATE_LOCK:
                    SYSTEM_STATE["demo_resets"] = resets
                consec_short = consec_short + 1 if steps_in_ep <= DEMO_SHORT_FALL_STEPS else 0
                if resets % DEMO_LOG_EVERY_N_RESETS == 0:
                    add_log(f"👁️ Demo reset x{resets} ({steps_in_ep} steps)")
                if resets >= DEMO_EXIT_AFTER_RESETS or consec_short >= DEMO_EXIT_AFTER_CONSEC_SHORT:
                    add_log("👁️ Demo unstable — returning to training")
                    break
                obs, prev_tau = env.reset()
                obs = obs.to(DEVICE)
                prev_action = torch.from_numpy(prev_tau / TAU_MAX).float().unsqueeze(0).to(DEVICE)
                h = agent.init_hidden(1).to(DEVICE)
                steps_in_ep = 0

            time.sleep(max(0, dt_target - (time.time() - t0)))

        with _STATE_LOCK:
            SYSTEM_STATE["mode"] = "TRAINING"
            SYSTEM_STATE["demo_stop"] = False
            SYSTEM_STATE["status"] = "🔥 BACK TO FORGING"

    def run(self):
        load_data()
        gen = 1
        with _STATE_LOCK:
            if SYSTEM_STATE["generation"] > 0:
                gen = SYSTEM_STATE["generation"] + 1
            bg = SYSTEM_STATE.get("best_genome")
            bw = SYSTEM_STATE.get("best_weights")

        if bg and len(bg) == GENOME_DIM:
            self.population_data[0] = {"genome": np.array(bg), "weights": bw}
            add_log("🧬 Injected saved champion")

        add_log(f"🧪 Evolution: pop={self.pop_size} workers={CORE_COUNT} elites={N_ELITES}")
        add_log("🎯 v2.2: Constraints + Diversity + Compound fitness")
        add_log(f"📐 Constraints: res≥{MIN_RESERVOIR_SIZE} dens≥{MIN_DENSITY:.0%} rad≤{MAX_SPECTRAL_RADIUS} alive≥{MIN_ALIVE_WEIGHT:.0%}")

        with ProcessPoolExecutor(max_workers=CORE_COUNT) as executor:
            while self.running:
                with _STATE_LOCK:
                    manual = SYSTEM_STATE["manual_demo_request"]
                    if manual:
                        SYSTEM_STATE["manual_demo_request"] = False
                        cg = SYSTEM_STATE.get("best_genome")
                        cw = SYSTEM_STATE.get("best_weights")
                        cs = SYSTEM_STATE.get("best_score", -1e9)

                if manual and cg:
                    add_log("▶️ User requested demo")
                    self.run_demo_mode({
                        "genome": cg,
                        "params": GeneDecoder.decode(np.array(cg)),
                        "weights": cw, "id": "CHAMPION", "fitness": cs
                    })

                while True:
                    with _STATE_LOCK:
                        if SYSTEM_STATE["mode"] != "DEMO":
                            break
                    time.sleep(0.25)

                with _STATE_LOCK:
                    SYSTEM_STATE["status"] = f"🔥 FORGING GEN {gen}"
                    SYSTEM_STATE["generation"] = gen
                    SYSTEM_STATE["gene_sigma"] = self.nge.sigma.tolist()
                    SYSTEM_STATE["gene_sensitivity"] = self.nge.sensitivity.tolist()
                    SYSTEM_STATE["stagnation_count"] = self.nge.stagnation_count

                seed_base = random.randint(1, 1_000_000)
                futures = [executor.submit(run_life_cycle, {
                    "genome": pd["genome"].tolist(),
                    "weights": pd.get("weights"),
                    "gen": gen, "seed": seed_base + i * 17
                }) for i, pd in enumerate(self.population_data)]

                results = []
                for f in as_completed(futures):
                    res = f.result()
                    if "error" in res:
                        add_log(f"⚠️ Error: {res['error'][:60]}")
                    else:
                        results.append(res)

                if not results:
                    add_log("❌ All failed. Reseeding.")
                    self.population_data = [{"genome": np.random.rand(GENOME_DIM), "weights": None}
                                            for _ in range(self.pop_size)]
                    continue

                self.nge.update_from_generation(results)

                results.sort(key=lambda r: (r["mean_steps"], r["mean_return"]), reverse=True)
                elites = results[:N_ELITES]
                best = results[0]

                pop_vecs = [list(r["genome"][:HYPER_D]) + [1.0 if i < N_ELITES else 0.0] 
                            for i, r in enumerate(results)]
                avg_elite = np.mean([np.array(e["genome"][:HYPER_D]) for e in elites], axis=0).tolist()

                with _STATE_LOCK:
                    SYSTEM_STATE["pop_vectors"] = pop_vecs
                    SYSTEM_STATE["history_vectors"].append(avg_elite)
                    if len(SYSTEM_STATE["history_vectors"]) > 160:
                        SYSTEM_STATE["history_vectors"].pop(0)
                    SYSTEM_STATE["hyperparams"] = best["params"]
                    SYSTEM_STATE["current_id"] = best["id"]
                    prior_best = SYSTEM_STATE["best_score"]

                # Check for stagnation
                self.nge.check_stagnation(best["mean_steps"])

                if best["mean_steps"] > prior_best:
                    with _STATE_LOCK:
                        SYSTEM_STATE["best_score"] = best["mean_steps"]
                        SYSTEM_STATE["best_genome"] = best["genome"]
                        if best.get("weights"):
                            SYSTEM_STATE["best_weights"] = best["weights"]
                    save_data(best["mean_steps"], gen, best["genome"], best.get("weights"))
                    self.nge.save(SIGMA_FILE)
                    pools = f"F{best['params']['pool_fast']:.0%}/M{best['params']['pool_med']:.0%}/S{best['params']['pool_slow']:.0%}"
                    alive_w = best['params']['reward_w'][0]
                    add_log(f"🏆 NEW BEST: {best['id']} steps={best['mean_steps']:.1f} alive_w={alive_w:.2f} {pools}")
                else:
                    tag = " [T]" if best.get("titan") else (" [V]" if best.get("varsity") else "")
                    stag = f" stag={self.nge.stagnation_count}" if self.nge.stagnation_count > 5 else ""
                    add_log(f"Gen {gen}: {best['id']}{tag} steps={best['mean_steps']:.1f}{stag}")

                if gen % 10 == 0:
                    add_log(self.nge.get_sigma_summary())

                if best["mean_steps"] >= AUTO_DEMO_STEPS:
                    add_log("👁️ Auto-demo triggered")
                    self.run_demo_mode(best)

                # Reproduce with diversity injection
                self.population_data = self.nge.reproduce(elites, self.pop_size, gen)
                gen += 1


# =============================================================================
# WEB UI
# =============================================================================

HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>🔥 Pendulum Furnace v2.2 CONSTRAINED</title>
  <style>
    :root{ --bg:#05070c; --panel:#0b0f14; --panel2:#0f1720; --stroke:#223044; --text:#e6eef8; --muted:#b8c7dd; --c1:#ff9d00; --c2:#ff5500; --ng:#22d3ee; --cross:#a855f7; --warn:#fbbf24; }
    body { margin:0; background:var(--bg); color:var(--text); font-family: ui-sans-serif, system-ui, sans-serif; }
    .wrap { display:flex; height:100vh; }
    aside { width:440px; background:var(--panel); border-right:1px solid var(--stroke); padding:16px; display:flex; flex-direction:column; gap:12px; overflow:auto; }
    main { flex:1; padding:16px; display:grid; grid-template-columns:1fr 1fr; grid-template-rows:2fr 1fr; gap:12px; }
    h1 { font-size:16px; margin:0; }
    .card { background:var(--panel2); border:1px solid var(--stroke); border-radius:14px; padding:12px; }
    .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    button { padding:10px 12px; border-radius:12px; border:1px solid var(--stroke); background:#132034; color:var(--text); cursor:pointer; }
    button:hover { background:#182a45; }
    button.active { background:#2d4a6f; border-color:#4a7ab0; }
    .stat { display:flex; justify-content:space-between; font-size:12px; color:var(--muted); margin-bottom:6px; }
    .val { color:var(--text); font-weight:700; }
    .mono { font-family: ui-monospace, monospace; font-size:11px; white-space:pre-wrap; }
    canvas { width:100%; height:100%; display:block; background:var(--panel2); border:1px solid var(--stroke); border-radius:14px; }
    .label { position:absolute; margin:10px; font-size:11px; color:#7f97b6; }
    .box { position:relative; }
    .tag { display:inline-block; padding:2px 6px; border-radius:4px; font-size:10px; margin-right:4px; }
    .tag-fast { background:#ef4444; color:white; }
    .tag-med { background:#f59e0b; color:black; }
    .tag-slow { background:#3b82f6; color:white; }
    .ng { color:#22d3ee; font-weight:600; }
    .cross { color:#a855f7; font-weight:600; }
    .sigma-bar { height:8px; background:#1e293b; border-radius:4px; margin:2px 0; }
    .sigma-fill { height:100%; border-radius:4px; transition:width 0.3s; }
    .constraint { color:#fbbf24; font-size:10px; }
  </style>
</head>
<body>
<div class="wrap">
  <aside>
    <h1>🔥 Pendulum Furnace v2.2</h1>
    <div style="font-size:11px;margin-top:-8px;">
      <span class="ng">CONSTRAINED</span> + <span class="cross">DIVERSITY INJECTION</span>
    </div>
    <div class="card">
      <div class="stat"><span>STATUS</span><span class="val" id="status">—</span></div>
      <div class="stat"><span>MODE</span><span class="val" id="mode">—</span></div>
      <div class="stat"><span>GEN</span><span class="val" id="gen">0</span></div>
      <div class="stat"><span>BEST (STEPS)</span><span class="val" id="best" style="color:var(--c2)">—</span></div>
      <div class="stat"><span>STAGNATION</span><span class="val" id="stag">0</span></div>
      <div class="stat"><span>CURRENT</span><span class="val" id="cid">—</span></div>
      <div class="stat"><span>RESERVOIR</span><span class="val" id="nres">—</span></div>
      <div class="stat"><span>POOLS</span><span class="val" id="pools">—</span></div>
      <div class="stat"><span>EPISODE STEP</span><span class="val" id="estep">—</span></div>
    </div>
    <div class="card">
      <div class="row">
        <button onclick="triggerDemo()">▶ Demo</button>
        <button onclick="stopDemo()">⏹ Stop</button>
        <button id="btnHyper" class="active" onclick="setViz('hyperspace')">🌌 Hyper</button>
        <button id="btnNeural" onclick="setViz('neural')">🧠 Neural</button>
        <button id="btnSigma" onclick="setViz('sigma')">📊 Sigma</button>
      </div>
      <div style="height:10px"></div>
      <div class="mono" id="hp">hyperparams…</div>
      <div style="height:8px"></div>
      <div class="mono" id="rw">reward weights…</div>
    </div>
    <div class="card">
      <div style="font-size:12px;color:var(--muted);margin-bottom:6px;"><b>Adaptive σ (per-gene mutation)</b></div>
      <div id="sigma-bars"></div>
    </div>
    <div class="card">
      <div style="font-size:12px;color:var(--muted);margin-bottom:6px;"><b>Logs</b></div>
      <div class="mono" id="logs">…</div>
    </div>
    <div class="card" style="font-size:11px;color:var(--muted);">
      <b>v2.2 Constraints:</b><br/>
      <span class="constraint">📐 reservoir ≥ 400 | density ≥ 8% | radius ≤ 1.5</span><br/>
      <span class="constraint">💚 alive_weight ≥ 15% (must care about survival!)</span><br/>
      🌱 Diversity injection every 50 gens<br/>
      ⚡ Sigma boost after 30 stagnant gens<br/>
      📊 Compound fitness: steps × (1 + 0.1×return)
    </div>
  </aside>
  <main>
    <div class="box"><div class="label">Triple Pendulum (ground contact fail)</div><canvas id="sim" width="920" height="920"></canvas></div>
    <div class="box"><div class="label" id="vizLabel">Hyperspace</div><canvas id="map" width="920" height="920"></canvas></div>
    <div class="box" style="grid-column:span 2;"><div class="label">Telemetry</div><canvas id="tele" width="1880" height="340"></canvas></div>
  </main>
</div>
<script>
const sim=document.getElementById("sim"),simc=sim.getContext("2d");
const map=document.getElementById("map"),mapc=map.getContext("2d");
const tele=document.getElementById("tele"),telec=tele.getContext("2d");
let vizMode="hyperspace";
const GENE_NAMES=["SIZE","DENS","LK_F","LK_M","LK_S","PL_F","PL_S","RAD","LR","W_ALV","W_TIP","W_UP","W_VEL","W_TAU"];

function setViz(m){vizMode=m;
  document.getElementById("btnHyper").classList.toggle("active",m==="hyperspace");
  document.getElementById("btnNeural").classList.toggle("active",m==="neural");
  document.getElementById("btnSigma").classList.toggle("active",m==="sigma");
  document.getElementById("vizLabel").textContent=m==="hyperspace"?"Hyperspace":(m==="neural"?"Neural Net":"Gene σ Adaptation");
  fetch("/set_viz_mode",{method:"POST",body:m});
}

function w2c(x,y,reach,W,H){const s=Math.min(W,H)*0.32/(reach+1e-9);return[W/2+x*s,H*0.75-y*s];}

function drawSim(s){
  const W=sim.width,H=sim.height;simc.clearRect(0,0,W,H);
  const reach=s.reach||2.4;
  
  const[gx1,gy1]=w2c(-reach*2,0,reach,W,H);
  const[gx2,gy2]=w2c(reach*2,0,reach,W,H);
  simc.strokeStyle="#ef4444";simc.lineWidth=3;simc.setLineDash([8,4]);
  simc.beginPath();simc.moveTo(gx1,gy1);simc.lineTo(gx2,gy2);simc.stroke();
  simc.setLineDash([]);
  
  simc.fillStyle="rgba(239,68,68,0.1)";
  simc.fillRect(0,gy1,W,H-gy1);
  
  simc.fillStyle="#ef4444";simc.font="12px monospace";
  simc.fillText("GROUND (y=0)",gx2-120,gy1-8);
  
  const pts=s.points||[];if(!pts.length)return;
  
  simc.lineWidth=10;simc.lineCap="round";simc.strokeStyle="#dbeafe";simc.beginPath();
  pts.forEach(([x,y],i)=>{const[px,py]=w2c(x,y,reach,W,H);i===0?simc.moveTo(px,py):simc.lineTo(px,py);});
  simc.stroke();
  
  pts.forEach(([x,y],i)=>{
    const[px,py]=w2c(x,y,reach,W,H);
    const danger = y <= 0.3;
    simc.fillStyle = i===0 ? "#64748b" : (danger ? "#fbbf24" : "#93c5fd");
    simc.beginPath();simc.arc(px,py,i===0?6:8,0,Math.PI*2);simc.fill();
  });
  
  if(s.tip){
    const[tx,ty]=s.tip,[tpx,tpy]=w2c(tx,ty,reach,W,H);
    simc.fillStyle="#fbbf24";simc.beginPath();simc.arc(tpx,tpy,6,0,Math.PI*2);simc.fill();
    simc.fillStyle="#b8c7dd";simc.font="11px monospace";
    simc.fillText(`tip_y: ${ty.toFixed(2)}`,tpx+12,tpy);
  }
}

const LABELS=["SIZE","DENS","LK_F","LK_M","LK_S","PL_F","PL_S","RAD","LR"];
let axisX=0,axisY=7,lastSwitch=Date.now();

function drawHyper(pop,hist){
  const W=map.width,H=map.height;mapc.clearRect(0,0,W,H);
  mapc.strokeStyle="#223044";mapc.lineWidth=2;
  mapc.beginPath();mapc.moveTo(0,H/2);mapc.lineTo(W,H/2);mapc.stroke();
  mapc.beginPath();mapc.moveTo(W/2,0);mapc.lineTo(W/2,H);mapc.stroke();
  mapc.fillStyle="#b8c7dd";mapc.font="14px monospace";
  mapc.fillText(`X:${LABELS[axisX]}`,W-100,H-16);mapc.fillText(`Y:${LABELS[axisY]}`,16,28);
  const proj=v=>[60+v[axisX]*(W-120),(H-60)-v[axisY]*(H-120)];
  if(hist&&hist.length>1){mapc.strokeStyle="rgba(34,211,238,0.6)";mapc.lineWidth=3;mapc.beginPath();hist.forEach((v,i)=>{const[x,y]=proj(v);i===0?mapc.moveTo(x,y):mapc.lineTo(x,y);});mapc.stroke();const[lx,ly]=proj(hist[hist.length-1]);mapc.fillStyle="#22d3ee";mapc.beginPath();mapc.arc(lx,ly,6,0,Math.PI*2);mapc.fill();}
  if(pop)pop.forEach((v,i)=>{const elite=v[9]>0.5,[x,y]=proj(v);mapc.beginPath();mapc.arc(x,y,elite?7:3,0,Math.PI*2);mapc.fillStyle=elite?"#ff5500":"#44556c";mapc.fill();});
  if(Date.now()-lastSwitch>3500){lastSwitch=Date.now();axisX=Math.floor(Math.random()*9);do{axisY=Math.floor(Math.random()*9);}while(axisY===axisX);}
}

function drawNeural(brain){
  const W=map.width,H=map.height;mapc.clearRect(0,0,W,H);
  if(!brain||!brain.activations){mapc.fillStyle="#556";mapc.font="16px sans-serif";mapc.fillText("Waiting for demo…",W/2-80,H/2);return;}
  const acts=brain.activations,links=brain.links||[],lw=brain.link_weights||[],nRes=brain.n_reservoir||acts.length,nViz=acts.length;
  const cx=W/2,cy=H/2,rad=Math.min(W,H)*0.36;
  const nPos=i=>{const th=(i/nViz)*2*Math.PI-Math.PI/2;return[cx+rad*Math.cos(th),cy+rad*Math.sin(th)];};
  const stride=Math.max(1,Math.floor(nRes/nViz));
  const poolInfo=brain.pool_info||{pool_boundaries:[0,nViz/3,2*nViz/3,nViz]};
  const pb=poolInfo.pool_boundaries||[0,nViz/3,2*nViz/3,nViz];
  const getPoolColor=(idx)=>{const ri=idx*stride;if(ri<pb[1])return"#ef4444";if(ri<pb[2])return"#f59e0b";return"#3b82f6";};
  mapc.lineWidth=0.5;
  for(let i=0;i<Math.min(links.length,300);i++){
    const[src,dst]=links[i],si=Math.floor(src/stride),di=Math.floor(dst/stride);
    if(si>=nViz||di>=nViz)continue;
    const[x1,y1]=nPos(si),[x2,y2]=nPos(di),w=lw[i]||0,flow=Math.abs((acts[si]||0)*w),alpha=Math.min(0.8,flow*2+0.05);
    mapc.strokeStyle=w>0?`rgba(34,211,238,${alpha})`:`rgba(255,100,100,${alpha})`;
    mapc.beginPath();mapc.moveTo(x1,y1);mapc.lineTo(x2,y2);mapc.stroke();
  }
  for(let i=0;i<nViz;i++){
    const[x,y]=nPos(i),act=acts[i]||0,r=2+Math.abs(act)*5;
    mapc.fillStyle=getPoolColor(i);mapc.globalAlpha=0.3+Math.abs(act)*0.7;
    mapc.beginPath();mapc.arc(x,y,r,0,Math.PI*2);mapc.fill();
  }
  mapc.globalAlpha=1;
  if(brain.prev_action){mapc.fillStyle="#22d3ee";mapc.font="11px monospace";mapc.fillText(`feedback τ: [${brain.prev_action.map(x=>x.toFixed(1)).join(",")}]`,10,H-30);}
  if(brain.mu){mapc.fillStyle="#fbbf24";mapc.font="12px monospace";mapc.fillText(`τ:[${brain.mu.map(x=>x.toFixed(2)).join(",")}]`,cx-60,cy-10);mapc.fillText(`V:${brain.v.toFixed(3)}`,cx-30,cy+10);}
  mapc.fillStyle="#778";mapc.font="11px monospace";mapc.fillText(`${nViz}/${nRes} neurons`,10,H-10);
}

function drawSigmaViz(sigma,sensitivity){
  const W=map.width,H=map.height;mapc.clearRect(0,0,W,H);
  if(!sigma||sigma.length===0){mapc.fillStyle="#556";mapc.font="16px sans-serif";mapc.fillText("No sigma data yet…",W/2-80,H/2);return;}
  
  const n=sigma.length;
  const barW=(W-100)/n-4;
  const maxSig=0.2,maxSens=1.0;
  
  mapc.fillStyle="#22d3ee";mapc.font="14px sans-serif";
  mapc.fillText("Per-Gene Mutation σ (cyan) vs Sensitivity (red)",20,30);
  mapc.fillStyle="#778";mapc.font="11px monospace";
  mapc.fillText("↑ High σ = explore more | High sensitivity = careful steps ↓",20,50);
  
  for(let i=0;i<n;i++){
    const x=50+i*(barW+4);
    const sigH=Math.min(1,sigma[i]/maxSig)*(H-140);
    const sensH=sensitivity?Math.min(1,sensitivity[i]/maxSens)*(H-140):0;
    
    mapc.fillStyle="rgba(34,211,238,0.8)";
    mapc.fillRect(x,H-80-sigH,barW/2-1,sigH);
    
    mapc.fillStyle="rgba(239,68,68,0.6)";
    mapc.fillRect(x+barW/2,H-80-sensH,barW/2-1,sensH);
    
    mapc.fillStyle="#b8c7dd";mapc.font="9px monospace";
    mapc.save();
    mapc.translate(x+barW/2,H-60);
    mapc.rotate(-Math.PI/4);
    mapc.fillText(GENE_NAMES[i],0,0);
    mapc.restore();
  }
  
  mapc.fillStyle="#22d3ee";mapc.fillRect(W-150,70,12,12);
  mapc.fillStyle="#b8c7dd";mapc.font="11px sans-serif";mapc.fillText("σ (mutation size)",W-130,81);
  mapc.fillStyle="#ef4444";mapc.fillRect(W-150,90,12,12);
  mapc.fillText("sensitivity",W-130,101);
}

function drawMap(pop,hist,brain,sigma,sensitivity){
  if(vizMode==="sigma")drawSigmaViz(sigma,sensitivity);
  else if(vizMode==="neural")drawNeural(brain);
  else drawHyper(pop,hist);
}

let teleHist=[];
function drawTele(s,brain){
  const W=tele.width,H=tele.height;telec.clearRect(0,0,W,H);
  telec.strokeStyle="#223044";telec.lineWidth=2;telec.strokeRect(1,1,W-2,H-2);
  if(!s||s.t===undefined)return;
  const v=brain?.v||0;
  teleHist.push([s.t,s.r||0,s.tip?s.tip[1]:0,v]);
  if(teleHist.length>240)teleHist.shift();
  const t0=teleHist[0][0],t1=teleHist[teleHist.length-1][0],dt=Math.max(1e-6,t1-t0);
  const xOf=t=>20+(t-t0)/dt*(W-40);
  const yOf=(val,vmin,vmax)=>(H-20)-(val-vmin)/(vmax-vmin+1e-9)*(H-40);
  const rV=teleHist.map(x=>x[1]),rmin=Math.min(...rV,-12),rmax=Math.max(...rV,6);
  telec.strokeStyle="rgba(255,157,0,0.9)";telec.lineWidth=2;telec.beginPath();
  teleHist.forEach((v,i)=>{const x=xOf(v[0]),y=yOf(v[1],rmin,rmax);i===0?telec.moveTo(x,y):telec.lineTo(x,y);});telec.stroke();
  const tyV=teleHist.map(x=>x[2]),tymax=Math.max(...tyV,2.4);
  telec.strokeStyle="rgba(147,197,253,0.9)";telec.beginPath();
  teleHist.forEach((v,i)=>{const x=xOf(v[0]),y=yOf(v[2],0,tymax);i===0?telec.moveTo(x,y):telec.lineTo(x,y);});telec.stroke();
  
  const groundY=yOf(0,0,tymax);
  telec.strokeStyle="rgba(239,68,68,0.5)";telec.setLineDash([4,4]);
  telec.beginPath();telec.moveTo(20,groundY);telec.lineTo(W-20,groundY);telec.stroke();
  telec.setLineDash([]);
  
  const vV=teleHist.map(x=>x[3]),vmin=Math.min(...vV,-5),vmax=Math.max(...vV,10);
  telec.strokeStyle="rgba(74,222,128,0.7)";telec.beginPath();
  teleHist.forEach((v,i)=>{const x=xOf(v[0]),y=yOf(v[3],vmin,vmax);i===0?telec.moveTo(x,y):telec.lineTo(x,y);});telec.stroke();
}

function updateSigmaBars(sigma,sensitivity){
  const container=document.getElementById("sigma-bars");
  if(!sigma||sigma.length===0){container.innerHTML="<span style='color:#556'>No data yet</span>";return;}
  let html="";
  for(let i=0;i<sigma.length;i++){
    const pct=Math.min(100,sigma[i]/0.2*100);
    const sens=sensitivity?sensitivity[i]:0;
    const color=sens>0.5?"#ef4444":"#22d3ee";
    html+=`<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">
      <span style="width:40px;font-size:9px;color:#778;">${GENE_NAMES[i]}</span>
      <div class="sigma-bar" style="flex:1;"><div class="sigma-fill" style="width:${pct}%;background:${color};"></div></div>
      <span style="width:35px;font-size:9px;color:#778;">${sigma[i].toFixed(3)}</span>
    </div>`;
  }
  container.innerHTML=html;
}

async function triggerDemo(){await fetch("/trigger_demo",{method:"POST"});}
async function stopDemo(){await fetch("/stop_demo",{method:"POST"});}

async function poll(){
  const r=await fetch("/status"),d=await r.json();
  document.getElementById("status").textContent=d.status;
  document.getElementById("mode").textContent=d.mode;
  document.getElementById("gen").textContent=d.gen;
  document.getElementById("best").textContent=(d.score||0).toFixed(1);
  document.getElementById("stag").textContent=d.stagnation_count||0;
  document.getElementById("cid").textContent=d.id;
  document.getElementById("nres").textContent=d.brain?.n_reservoir||(d.params?.n_reservoir||"—");
  document.getElementById("estep").textContent=d.sim?.steps||"—";
  const hp=d.params||{};
  const pf=(hp.pool_fast*100||0).toFixed(0),pm=(hp.pool_med*100||0).toFixed(0),ps=(hp.pool_slow*100||0).toFixed(0);
  document.getElementById("pools").innerHTML=`<span class="tag tag-fast">${pf}%</span><span class="tag tag-med">${pm}%</span><span class="tag tag-slow">${ps}%</span>`;
  document.getElementById("hp").textContent=`n_reservoir = ${hp.n_reservoir||"—"}\ndensity     = ${hp.density?.toFixed(4)||"—"}\nleak_fast   = ${hp.leak_fast?.toFixed(3)||"—"}\nleak_med    = ${hp.leak_med?.toFixed(3)||"—"}\nleak_slow   = ${hp.leak_slow?.toFixed(3)||"—"}\nspect_rad   = ${hp.spectral_radius?.toFixed(3)||"—"}\nlr          = ${hp.lr?.toExponential(2)||"—"}`;
  const rw=hp.reward_w||[0,0,0,0,0];
  const aliveColor = rw[0] >= 0.15 ? "#22c55e" : "#ef4444";
  document.getElementById("rw").innerHTML=`reward_w: <span style="color:${aliveColor}">alive=${rw[0]?.toFixed(2)}</span> tip=${rw[1]?.toFixed(2)} up=${rw[2]?.toFixed(2)} vel=${rw[3]?.toFixed(2)} tau=${rw[4]?.toFixed(2)}`;
  document.getElementById("logs").textContent=(d.logs||[]).slice(0,14).join("\n");
  
  const sigma=d.gene_sigma||[];
  const sensitivity=d.gene_sensitivity||[];
  updateSigmaBars(sigma,sensitivity);
  
  drawSim(d.sim||{});
  drawMap(d.pop_vectors||[],d.history_vectors||[],d.brain||{},sigma,sensitivity);
  drawTele(d.sim||{},d.brain||{});
}
setInterval(poll,90);poll();
</script>
</body>
</html>
"""

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

@app.get("/")
def index(): return Response(HTML, mimetype="text/html")

@app.get("/status")
def status():
    with _STATE_LOCK:
        return jsonify({
            "status": SYSTEM_STATE["status"], "gen": SYSTEM_STATE["generation"],
            "score": SYSTEM_STATE["best_score"], "mode": SYSTEM_STATE["mode"],
            "id": SYSTEM_STATE["current_id"], "logs": SYSTEM_STATE["logs"],
            "params": SYSTEM_STATE.get("hyperparams", {}),
            "pop_vectors": SYSTEM_STATE.get("pop_vectors", []),
            "history_vectors": SYSTEM_STATE.get("history_vectors", []),
            "sim": SYSTEM_STATE.get("sim_view", {}),
            "brain": SYSTEM_STATE.get("brain_view", {}),
            "demo_resets": SYSTEM_STATE.get("demo_resets", 0),
            "gene_sigma": SYSTEM_STATE.get("gene_sigma", []),
            "gene_sensitivity": SYSTEM_STATE.get("gene_sensitivity", []),
            "stagnation_count": SYSTEM_STATE.get("stagnation_count", 0),
        })

@app.post("/trigger_demo")
def trigger_demo():
    with _STATE_LOCK:
        if not SYSTEM_STATE["best_genome"]: return jsonify({"error": "No champion"}), 400
        if SYSTEM_STATE["mode"] == "DEMO": return jsonify({"error": "Already running"}), 400
        SYSTEM_STATE["manual_demo_request"] = True
    return jsonify({"status": "ok"})

@app.post("/stop_demo")
def stop_demo():
    with _STATE_LOCK:
        SYSTEM_STATE["demo_stop"] = True
        SYSTEM_STATE["mode"] = "TRAINING"
    add_log("⏹ Demo stopped")
    return jsonify({"status": "ok"})

@app.post("/set_viz_mode")
def set_viz_mode():
    from flask import request
    with _STATE_LOCK:
        SYSTEM_STATE["viz_mode"] = request.get_data(as_text=True) or "hyperspace"
    return jsonify({"status": "ok"})


def main():
    torch.set_num_threads(1)
    add_log("🔥 PENDULUM FURNACE v2.2 — CONSTRAINED + DIVERSITY")
    add_log(f"📐 Constraints: res≥{MIN_RESERVOIR_SIZE} dens≥{MIN_DENSITY:.0%} rad≤{MAX_SPECTRAL_RADIUS}")
    add_log(f"💚 alive_weight ≥ {MIN_ALIVE_WEIGHT:.0%} (must care about survival!)")
    add_log("🌱 Diversity injection every 50 gens | ⚡ Sigma boost on stagnation")
    add_log(f"🧪 Pop: {POP_SIZE} | Workers: {CORE_COUNT} | Elites: {N_ELITES}")
    add_log("🌐 http://127.0.0.1:5000")

    engine = EvolutionEngine()
    engine.start()

    with _STATE_LOCK:
        SYSTEM_STATE["status"] = "READY"

    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except:
        pass
    main()
