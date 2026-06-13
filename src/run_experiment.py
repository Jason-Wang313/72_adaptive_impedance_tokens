from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


BASE_SEED = 235725184
SEEDS = list(range(7))
EVAL_EPISODES = int(os.getenv("PAPER72_EVAL_EPISODES", "12"))
ABLATION_EPISODES = int(os.getenv("PAPER72_ABLATION_EPISODES", "10"))
STRESS_EPISODES = int(os.getenv("PAPER72_STRESS_EPISODES", "8"))
TRAINING_EXAMPLES = int(os.getenv("PAPER72_TRAINING_EXAMPLES", "2600"))
STEPS = 120
DT = 0.025
WALL_X = 0.0
Y_START = -0.43
Y_GOAL = 0.43
MAX_PENETRATION = 0.070

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


MODEL_XML = """
<mujoco model="adaptive_impedance_tokens">
  <compiler angle="radian"/>
  <option timestep="0.025" integrator="implicitfast" gravity="0 0 0"/>
  <default>
    <joint damping="0.18"/>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <geom name="table" type="plane" pos="0 0 -0.01" size="0.8 0.8 0.02"
          rgba="0.91 0.90 0.86 1"/>
    <geom name="surface_marker" type="box" pos="0 0 0.02" size="0.006 0.55 0.035"
          rgba="0.25 0.34 0.40 0.35"/>
    <body name="tool" pos="-0.11 -0.43 0.05">
      <joint name="tool_x" type="slide" axis="1 0 0" range="-0.18 0.13" damping="0.10"/>
      <joint name="tool_y" type="slide" axis="0 1 0" range="-0.55 0.55" damping="0.10"/>
      <geom name="tool_tip" type="sphere" size="0.035" mass="1.20" rgba="0.08 0.08 0.10 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="x_motor" joint="tool_x" gear="1" ctrllimited="true" ctrlrange="-65 65"/>
    <motor name="y_motor" joint="tool_y" gear="1" ctrllimited="true" ctrlrange="-65 65"/>
  </actuator>
</mujoco>
"""


METHODS = [
    "fixed_impedance",
    "gain_scheduled_impedance",
    "adaptive_impedance_control",
    "admittance_switching_control",
    "robust_mpc_impedance",
    "learned_gain_regressor",
    "impedance_token_policy",
    "oracle_impedance",
]

ABLATION_METHODS = [
    "token_full",
    "token_no_memory",
    "token_no_discrete_tokens",
    "token_no_force_update",
    "token_no_transition_planner",
    "token_no_safety_penalty",
]

STRESS_METHODS = [
    "adaptive_impedance_control",
    "admittance_switching_control",
    "robust_mpc_impedance",
    "learned_gain_regressor",
    "impedance_token_policy",
    "oracle_impedance",
]


@dataclass(frozen=True)
class SplitSpec:
    name: str
    stiffness: float
    stiffness_shift: float
    damping: float
    friction: float
    friction_shift: float
    target_force: float
    target_force_shift: float
    force_noise: float
    actuator_limit: float
    transition_bias: float


@dataclass(frozen=True)
class EpisodeConfig:
    split: SplitSpec
    seed: int
    episode: int
    stiffness: float
    stiffness_after: float
    damping: float
    friction: float
    friction_after: float
    target_force: float
    target_force_after: float
    force_noise: float
    actuator_limit: float
    shift_step: int
    target_shift_step: int
    stress_level: float | None = None


@dataclass
class MethodState:
    method: str
    k_est: float
    mu_est: float
    target_force_est: float
    token_scores: np.ndarray
    last_force_error: float
    last_force: float
    last_penetration: float
    force_error_integral: float
    chatter_crossings: int
    selected_token: int
    force_history: List[float]
    token_history: List[int]


@dataclass
class LearnedPack:
    scaler_x: StandardScaler
    scaler_y: StandardScaler
    model: Ridge
    training_rows: List[Dict[str, str]]
    train_mae: float


SPLITS = [
    SplitSpec("nominal_surface_tracking", 420.0, 1.00, 18.0, 0.18, 1.00, 12.0, 1.00, 0.28, 4.0, 0.20),
    SplitSpec("stiffness_shift", 360.0, 1.80, 19.0, 0.20, 1.00, 12.0, 1.00, 0.35, 3.6, 0.35),
    SplitSpec("friction_slip_shift", 430.0, 1.05, 16.0, 0.16, 2.25, 12.0, 1.00, 0.38, 3.5, 0.30),
    SplitSpec("contact_transition", 390.0, 1.35, 20.0, 0.22, 1.50, 11.0, 1.30, 0.42, 3.25, 0.65),
    SplitSpec("combined_stress", 340.0, 2.20, 17.0, 0.18, 2.60, 10.5, 1.55, 0.62, 2.85, 0.85),
]

TOKEN_TABLE = np.array(
    [
        [60.0, 9.0, 0.85, 7.0, 1.10],
        [90.0, 11.0, 1.00, 8.5, 1.00],
        [135.0, 15.0, 1.05, 9.0, 0.90],
        [115.0, 22.0, 0.92, 6.5, 1.35],
        [160.0, 24.0, 0.82, 5.5, 1.70],
        [100.0, 18.0, 1.25, 7.5, 1.15],
    ],
    dtype=float,
)


def ci95(values: Sequence[float]) -> float:
    vals = np.array(values, dtype=float)
    if len(vals) <= 1:
        return 0.0
    return float(1.96 * np.std(vals, ddof=1) / math.sqrt(len(vals)))


def make_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(MODEL_XML)


def config_rng(seed: int, episode: int, split_name: str) -> np.random.Generator:
    offset = sum((i + 1) * ord(c) for i, c in enumerate(split_name))
    return np.random.default_rng(BASE_SEED + 7919 * seed + 149 * episode + offset)


def make_config(split: SplitSpec, seed: int, episode: int, stress_level: float | None = None) -> EpisodeConfig:
    rng = config_rng(seed, episode, split.name if stress_level is None else f"{split.name}_{stress_level:.2f}")
    jitter = rng.normal(1.0, 0.08)
    if stress_level is None:
        stiffness = split.stiffness * jitter
        stiffness_after = split.stiffness * split.stiffness_shift * rng.normal(1.0, 0.06)
        friction = split.friction * rng.normal(1.0, 0.08)
        friction_after = split.friction * split.friction_shift * rng.normal(1.0, 0.08)
        target = split.target_force * rng.normal(1.0, 0.05)
        target_after = split.target_force * split.target_force_shift * rng.normal(1.0, 0.05)
        noise = split.force_noise
        actuator_limit = split.actuator_limit
    else:
        stiffness = (370.0 + 50.0 * rng.normal()) * (1.0 + 0.20 * stress_level)
        stiffness_after = stiffness * (1.0 + 1.45 * stress_level)
        friction = 0.16 + 0.05 * rng.random()
        friction_after = friction * (1.0 + 1.85 * stress_level)
        target = 10.5 + 2.0 * rng.random()
        target_after = target * (1.0 + 0.65 * stress_level)
        noise = 0.24 + 0.58 * stress_level
        actuator_limit = 4.0 - 1.20 * stress_level
    shift_step = int(rng.integers(42, 62))
    target_shift_step = int(rng.integers(58, 76))
    return EpisodeConfig(
        split=split,
        seed=seed,
        episode=episode,
        stiffness=max(160.0, stiffness),
        stiffness_after=max(180.0, stiffness_after),
        damping=split.damping * rng.normal(1.0, 0.07),
        friction=max(0.05, friction),
        friction_after=max(0.06, friction_after),
        target_force=max(6.0, target),
        target_force_after=max(6.0, target_after),
        force_noise=noise,
        actuator_limit=actuator_limit,
        shift_step=shift_step,
        target_shift_step=target_shift_step,
        stress_level=stress_level,
    )


def active_surface(cfg: EpisodeConfig, step: int) -> Tuple[float, float, float]:
    stiffness = cfg.stiffness_after if step >= cfg.shift_step else cfg.stiffness
    friction = cfg.friction_after if step >= cfg.shift_step else cfg.friction
    target = cfg.target_force_after if step >= cfg.target_shift_step else cfg.target_force
    return stiffness, friction, target


def init_state(method: str, cfg: EpisodeConfig) -> MethodState:
    scores = np.zeros(len(TOKEN_TABLE), dtype=float)
    scores[1] = 0.4
    return MethodState(
        method=method,
        k_est=420.0,
        mu_est=0.20,
        target_force_est=cfg.target_force,
        token_scores=scores,
        last_force_error=0.0,
        last_force=0.0,
        last_penetration=0.0,
        force_error_integral=0.0,
        chatter_crossings=0,
        selected_token=1,
        force_history=[],
        token_history=[],
    )


def surface_force(x: float, vx: float, vy: float, stiffness: float, damping: float, friction: float) -> Tuple[float, float, float]:
    penetration = max(0.0, x - WALL_X)
    if penetration <= 0.0:
        return 0.0, 0.0, 0.0
    normal = stiffness * penetration + damping * max(0.0, vx)
    tangent = -friction * normal * math.tanh(vy / 0.055)
    return normal, tangent, penetration


def train_feature(force: float, penetration: float, vx: float, vy: float, target_force: float, k_est: float, mu_est: float, phase: float, force_error: float) -> np.ndarray:
    safety_margin = max(0.0, penetration - 0.045)
    return np.array(
        [
            force,
            penetration,
            vx,
            vy,
            target_force,
            k_est,
            mu_est,
            phase,
            force_error,
            abs(force_error),
            force / max(1e-4, penetration + 0.005),
            abs(vy),
            safety_margin,
            force / max(1.0, target_force),
        ],
        dtype=float,
    )


def generate_training_pack() -> LearnedPack:
    rng = np.random.default_rng(BASE_SEED + 515)
    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    csv_rows: List[Dict[str, str]] = []
    for idx in range(TRAINING_EXAMPLES):
        stiffness = rng.uniform(220.0, 850.0)
        friction = rng.uniform(0.08, 0.72)
        target = rng.uniform(7.0, 20.0)
        penetration = rng.uniform(0.0, min(MAX_PENETRATION, target / stiffness * rng.uniform(0.45, 1.65)))
        vx = rng.normal(0.0, 0.22)
        vy = rng.normal(0.20, 0.32)
        normal = stiffness * penetration + rng.normal(0.0, 0.30)
        force_error = target - normal
        k_est = np.clip(normal / max(0.006, penetration), 180.0, 900.0) if penetration > 0.004 else 420.0
        mu_est = np.clip(friction + rng.normal(0.0, 0.08), 0.05, 0.80)
        phase = 1.0 if penetration > 0.005 else 0.0
        desired_pen = target / max(200.0, k_est)
        safety = max(0.0, penetration - 0.045)
        normal_kp = np.clip(75.0 + 0.16 * k_est + 3.0 * abs(force_error) - 1100.0 * safety, 45.0, 190.0)
        normal_kd = np.clip(8.0 + 0.030 * k_est + 10.0 * friction + 45.0 * safety, 7.0, 32.0)
        y_gain = np.clip(9.0 - 4.6 * friction - 45.0 * safety, 3.5, 10.5)
        target_scale = np.clip(desired_pen / max(1e-4, target / 420.0), 0.45, 1.85)
        feat = train_feature(normal, penetration, vx, vy, target, k_est, mu_est, phase, force_error)
        x_rows.append(feat)
        y_rows.append(np.array([normal_kp, normal_kd, y_gain, target_scale], dtype=float))
        csv_rows.append(
            {
                "example": str(idx),
                "stiffness": f"{stiffness:.4f}",
                "friction": f"{friction:.4f}",
                "target_force": f"{target:.4f}",
                "penetration": f"{penetration:.5f}",
                "force": f"{normal:.4f}",
                "normal_kp": f"{normal_kp:.4f}",
                "normal_kd": f"{normal_kd:.4f}",
                "y_gain": f"{y_gain:.4f}",
            }
        )
    x = np.vstack(x_rows)
    y = np.vstack(y_rows)
    scaler_x = StandardScaler().fit(x)
    scaler_y = StandardScaler().fit(y)
    model = Ridge(alpha=0.08)
    model.fit(scaler_x.transform(x), scaler_y.transform(y))
    pred = scaler_y.inverse_transform(model.predict(scaler_x.transform(x)))
    train_mae = float(np.mean(np.abs(pred - y)))
    return LearnedPack(scaler_x=scaler_x, scaler_y=scaler_y, model=model, training_rows=csv_rows, train_mae=train_mae)


def learned_gains(pack: LearnedPack, feat: np.ndarray) -> Tuple[float, float, float, float]:
    x_scaled = pack.scaler_x.transform(feat.reshape(1, -1))[0]
    pred_scaled = x_scaled @ pack.model.coef_.T + pack.model.intercept_
    pred = pred_scaled * pack.scaler_y.scale_ + pack.scaler_y.mean_
    return (
        float(np.clip(pred[0], 35.0, 210.0)),
        float(np.clip(pred[1], 5.0, 36.0)),
        float(np.clip(pred[2], 3.0, 11.5)),
        float(np.clip(pred[3], 0.40, 2.10)),
    )


def choose_token(state: MethodState, force: float, penetration: float, slip: float, target_force: float, method: str) -> int:
    scores = state.token_scores.copy()
    force_error = target_force - force
    if penetration < 0.006:
        scores += np.array([0.2, 0.5, 0.1, 0.8, 0.2, 0.0])
    if abs(force_error) > 3.0:
        scores += np.array([0.1, 0.2, 0.5, 0.1, -0.2, 0.6])
    if force > 1.45 * target_force or penetration > 0.045:
        scores += np.array([0.3, 0.1, -0.1, 0.7, 1.2, -0.2])
    if abs(slip) > 0.34:
        scores += np.array([0.0, 0.1, 0.2, 0.9, 0.8, -0.1])
    if method == "token_no_discrete_tokens":
        return 1
    return int(np.argmax(scores))


def controller_gains(
    method: str,
    state: MethodState,
    pack: LearnedPack,
    cfg: EpisodeConfig,
    step: int,
    x: float,
    y: float,
    vx: float,
    vy: float,
    force_obs: float,
    penetration: float,
    target_force: float,
    true_stiffness: float,
    true_friction: float,
) -> Tuple[float, float, float, float, int]:
    contact = penetration > 0.002
    k_obs = np.clip(force_obs / max(0.004, penetration), 170.0, 950.0) if contact else state.k_est
    if contact and method not in {"token_no_force_update"}:
        state.k_est = 0.90 * state.k_est + 0.10 * k_obs
        slip_mu = min(0.9, abs(vy) / max(0.02, force_obs + 1.0))
        state.mu_est = 0.94 * state.mu_est + 0.06 * max(0.05, min(0.75, true_friction + slip_mu))
    state.force_error_integral = np.clip(state.force_error_integral + (target_force - force_obs) * DT, -25.0, 25.0)
    force_error = target_force - force_obs
    phase = 1.0 if contact else 0.0

    token_idx = -1
    if method == "fixed_impedance":
        return 92.0, 11.0, 7.0, target_force / 420.0, token_idx
    if method == "gain_scheduled_impedance":
        kp = 72.0 + 4.0 * abs(force_error) + (24.0 if contact else 8.0)
        kd = 9.0 + 0.025 * state.k_est + (4.0 if force_obs > target_force else 0.0)
        y_gain = 8.5 - 2.8 * min(0.7, state.mu_est)
        return float(np.clip(kp, 55.0, 170.0)), float(np.clip(kd, 7.0, 28.0)), y_gain, target_force / max(220.0, state.k_est), token_idx
    if method == "adaptive_impedance_control":
        kp = 68.0 + 0.13 * state.k_est + 2.0 * abs(force_error)
        kd = 7.0 + 0.026 * state.k_est + 2.0 * max(0.0, force_obs / max(1.0, target_force) - 1.0)
        y_gain = 8.2 - 3.2 * min(0.8, state.mu_est)
        desired_pen = target_force / max(190.0, state.k_est) + 0.0008 * state.force_error_integral
        return float(np.clip(kp, 45.0, 185.0)), float(np.clip(kd, 7.0, 32.0)), float(np.clip(y_gain, 3.5, 10.0)), desired_pen, token_idx
    if method == "admittance_switching_control":
        if not contact:
            return 130.0, 12.0, 8.0, 0.020, token_idx
        kp = 78.0 + 2.5 * abs(force_error)
        kd = 18.0 + 0.014 * state.k_est
        y_gain = 5.2 if abs(force_error) > 4.0 else 7.0
        return float(np.clip(kp, 55.0, 150.0)), float(np.clip(kd, 12.0, 34.0)), y_gain, target_force / max(230.0, state.k_est), token_idx
    if method == "robust_mpc_impedance":
        kp = 80.0 + 0.08 * min(760.0, state.k_est)
        kd = 24.0 + 0.012 * state.k_est
        y_gain = 4.8
        desired_pen = 0.88 * target_force / max(260.0, state.k_est)
        return float(np.clip(kp, 65.0, 145.0)), float(np.clip(kd, 15.0, 36.0)), y_gain, desired_pen, token_idx
    if method == "learned_gain_regressor":
        feat = train_feature(force_obs, penetration, vx, vy, target_force, state.k_est, state.mu_est, phase, force_error)
        kp, kd, y_gain, scale = learned_gains(pack, feat)
        return kp, kd, y_gain, scale * target_force / 420.0, token_idx
    if method in {"impedance_token_policy", "token_full", "token_no_memory", "token_no_discrete_tokens", "token_no_force_update", "token_no_transition_planner", "token_no_safety_penalty"}:
        token_method = "impedance_token_policy" if method == "token_full" else method
        if method == "token_no_memory":
            state.token_scores[:] = 0.0
        token_idx = choose_token(state, force_obs, penetration, vy, target_force, method)
        token = TOKEN_TABLE[token_idx]
        if method != "token_no_memory":
            reward = -abs(force_error) / max(1.0, target_force) - 1.2 * max(0.0, penetration - 0.048) - 0.15 * abs(vy)
            state.token_scores[token_idx] = 0.92 * state.token_scores[token_idx] + 0.08 * reward
        kp = token[0] + 0.05 * state.k_est
        kd = token[1] + 0.010 * state.k_est
        y_gain = token[3] - 2.4 * min(0.8, state.mu_est)
        if method == "token_no_transition_planner" and not contact:
            y_gain += 2.0
        if method == "token_no_safety_penalty":
            desired_pen = token[2] * target_force / max(200.0, state.k_est)
        else:
            safety_scale = 0.82 if penetration > 0.050 or force_obs > 1.55 * target_force else token[2]
            desired_pen = safety_scale * target_force / max(210.0, state.k_est)
        return float(np.clip(kp, 45.0, 205.0)), float(np.clip(kd, 6.0, 36.0)), float(np.clip(y_gain, 2.8, 10.5)), desired_pen, token_idx
    if method == "oracle_impedance":
        kp = 72.0 + 0.15 * true_stiffness
        kd = 10.0 + 0.032 * true_stiffness + 8.0 * true_friction
        y_gain = 9.5 - 3.4 * min(0.75, true_friction)
        desired_pen = 1.05 * target_force / max(180.0, true_stiffness)
        return float(np.clip(kp, 55.0, 205.0)), float(np.clip(kd, 9.0, 36.0)), float(np.clip(y_gain, 4.0, 11.0)), desired_pen, token_idx
    raise ValueError(f"unknown method {method}")


def simulate_episode(model: mujoco.MjModel, method: str, cfg: EpisodeConfig, pack: LearnedPack) -> Dict[str, str]:
    rng = np.random.default_rng(BASE_SEED + 1019 * cfg.seed + 337 * cfg.episode + sum(ord(c) for c in method))
    data = mujoco.MjData(model)
    data.qpos[:2] = np.array([-0.115 + rng.normal(0.0, 0.006), Y_START + rng.normal(0.0, 0.008)])
    data.qvel[:2] = 0.0
    mujoco.mj_forward(model, data)
    state = init_state(method, cfg)

    force_errors: List[float] = []
    abs_force_errors: List[float] = []
    post_contact_errors: List[float] = []
    samples: List[str] = []
    contact_steps = 0
    safety_steps = 0
    slip_steps = 0
    chatter_steps = 0
    overshoot = 0.0
    max_penetration = 0.0
    energy = 0.0
    work = 0.0
    settled_step: int | None = None
    last_error_sign = 0
    final_progress = 0.0

    for step in range(STEPS):
        x, y = float(data.qpos[0]), float(data.qpos[1])
        vx, vy = float(data.qvel[0]), float(data.qvel[1])
        stiffness, friction, target_force = active_surface(cfg, step)
        normal_force, tangential_force, penetration = surface_force(x, vx, vy, stiffness, cfg.damping, friction)
        observed_force = max(0.0, normal_force + rng.normal(0.0, cfg.force_noise))
        force_error = target_force - observed_force
        state.force_history.append(observed_force)

        kp, kd, y_gain, desired_pen, token_idx = controller_gains(
            method, state, pack, cfg, step, x, y, vx, vy, observed_force, penetration, target_force, stiffness, friction
        )
        state.selected_token = token_idx
        state.token_history.append(token_idx)
        if abs(force_error) > 0.0:
            sign = 1 if force_error > 0 else -1
            if last_error_sign != 0 and sign != last_error_sign and abs(force_error) > 1.0 and penetration > 0.003:
                state.chatter_crossings += 1
                chatter_steps += 1
            last_error_sign = sign

        desired_x = WALL_X + float(np.clip(desired_pen, 0.004, MAX_PENETRATION))
        progress = min(1.0, step / (STEPS - 20))
        desired_y = Y_START + (Y_GOAL - Y_START) * progress
        if penetration < 0.004 and step < 30:
            desired_y = y
        # The benchmark uses an external compliant wall model rather than MuJoCo contact
        # pairs, so add explicit force-error feedback to keep sustained contact observable.
        normal_ctrl = kp * (desired_x - x) - kd * vx + 1.15 * force_error
        tangent_ctrl = 3.00 * y_gain * (desired_y - y) - 1.85 * vy

        data.qfrc_applied[:] = 0.0
        if penetration > 0.0:
            data.qfrc_applied[0] += -normal_force
            data.qfrc_applied[1] += tangential_force
            contact_steps += 1
            abs_force_errors.append(abs(force_error))
            if step > cfg.shift_step:
                post_contact_errors.append(abs(force_error))
            overshoot = max(overshoot, max(0.0, observed_force - target_force) / max(1.0, target_force))
            max_penetration = max(max_penetration, penetration)
            if abs(vy) > 0.42:
                slip_steps += 1
            if penetration > 0.060 or observed_force > 2.15 * target_force:
                safety_steps += 1
            if settled_step is None and step > cfg.target_shift_step + 4 and abs(force_error) / max(1.0, target_force) < 0.20:
                settled_step = step
        force_errors.append(force_error)

        ctrl = np.array([normal_ctrl, tangent_ctrl], dtype=float)
        ctrl = np.clip(ctrl, -65.0 * cfg.actuator_limit / 4.0, 65.0 * cfg.actuator_limit / 4.0)
        data.ctrl[:2] = ctrl
        v = np.array([vx, vy], dtype=float)
        energy += float(np.sum(np.abs(ctrl * v)) * DT)
        work += float(np.sum(ctrl * v) * DT)
        mujoco.mj_step(model, data)
        data.qpos[0] = float(np.clip(data.qpos[0], -0.18, 0.13))
        data.qpos[1] = float(np.clip(data.qpos[1], -0.55, 0.55))
        data.qvel[:] = np.clip(data.qvel[:], -1.8, 1.8)
        mujoco.mj_forward(model, data)
        final_progress = (float(data.qpos[1]) - Y_START) / (Y_GOAL - Y_START)

        if step % 15 == 0 or step == STEPS - 1:
            samples.append(
                f"{step}:x{x:.3f}:y{y:.3f}:f{observed_force:.2f}:tf{target_force:.2f}:p{penetration:.4f}:tok{token_idx}"
            )

    mean_abs_error = float(np.mean(abs_force_errors)) if abs_force_errors else cfg.target_force
    post_error = float(np.mean(post_contact_errors)) if post_contact_errors else mean_abs_error
    contact_ratio = contact_steps / STEPS
    safety_rate = safety_steps / max(1, contact_steps)
    slip_rate = slip_steps / max(1, contact_steps)
    chatter_rate = chatter_steps / max(1, contact_steps)
    settle_latency = STEPS if settled_step is None else max(0, settled_step - cfg.target_shift_step)
    normalized_error = mean_abs_error / max(1.0, cfg.target_force_after)
    success = int(
        contact_ratio > 0.45
        and final_progress > 0.78
        and normalized_error < 0.37
        and overshoot < 1.05
        and safety_rate < 0.12
        and slip_rate < 0.42
    )
    return {
        "method": method,
        "split": cfg.split.name,
        "seed": str(cfg.seed),
        "episode": str(cfg.episode),
        "stress_level": "" if cfg.stress_level is None else f"{cfg.stress_level:.2f}",
        "success": str(success),
        "contact_ratio": f"{contact_ratio:.5f}",
        "mean_abs_force_error": f"{mean_abs_error:.5f}",
        "post_shift_force_error": f"{post_error:.5f}",
        "normalized_force_error": f"{normalized_error:.5f}",
        "peak_overshoot": f"{overshoot:.5f}",
        "max_penetration": f"{max_penetration:.5f}",
        "safety_violation_rate": f"{safety_rate:.5f}",
        "slip_rate": f"{slip_rate:.5f}",
        "chatter_rate": f"{chatter_rate:.5f}",
        "chatter_crossings": str(state.chatter_crossings),
        "settling_latency": str(settle_latency),
        "energy": f"{energy:.5f}",
        "work": f"{work:.5f}",
        "final_progress": f"{final_progress:.5f}",
        "final_k_est": f"{state.k_est:.5f}",
        "final_mu_est": f"{state.mu_est:.5f}",
        "token_switches": str(sum(1 for a, b in zip(state.token_history, state.token_history[1:]) if a != b and a >= 0 and b >= 0)),
        "trajectory_samples": ";".join(samples),
    }


def group_rows(rows: Iterable[Dict[str, str]], fields: Sequence[str]) -> Dict[Tuple[str, ...], List[Dict[str, str]]]:
    grouped: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in fields), []).append(row)
    return grouped


def mean_metric(rows: Sequence[Dict[str, str]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def build_seed_metrics(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for (method, split, seed), group in sorted(group_rows(rows, ["method", "split", "seed"]).items()):
        out.append(
            {
                "method": method,
                "split": split,
                "seed": seed,
                "episodes": str(len(group)),
                "success_rate": f"{mean_metric(group, 'success'):.5f}",
                "mean_abs_force_error": f"{mean_metric(group, 'mean_abs_force_error'):.5f}",
                "post_shift_force_error": f"{mean_metric(group, 'post_shift_force_error'):.5f}",
                "normalized_force_error": f"{mean_metric(group, 'normalized_force_error'):.5f}",
                "peak_overshoot": f"{mean_metric(group, 'peak_overshoot'):.5f}",
                "safety_violation_rate": f"{mean_metric(group, 'safety_violation_rate'):.5f}",
                "slip_rate": f"{mean_metric(group, 'slip_rate'):.5f}",
                "chatter_rate": f"{mean_metric(group, 'chatter_rate'):.5f}",
                "mean_settling_latency": f"{mean_metric(group, 'settling_latency'):.5f}",
                "mean_energy": f"{mean_metric(group, 'energy'):.5f}",
                "mean_final_progress": f"{mean_metric(group, 'final_progress'):.5f}",
                "mean_token_switches": f"{mean_metric(group, 'token_switches'):.5f}",
            }
        )
    return out


def build_summary(seed_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    metrics = [
        "success_rate",
        "mean_abs_force_error",
        "post_shift_force_error",
        "normalized_force_error",
        "peak_overshoot",
        "safety_violation_rate",
        "slip_rate",
        "chatter_rate",
        "mean_settling_latency",
        "mean_energy",
        "mean_final_progress",
        "mean_token_switches",
    ]
    rows: List[Dict[str, str]] = []
    for (method, split), group in sorted(group_rows(seed_rows, ["method", "split"]).items()):
        item: Dict[str, str] = {"method": method, "split": split, "seeds": str(len(group)), "episodes_per_seed": group[0]["episodes"]}
        for metric in metrics:
            vals = [float(row[metric]) for row in group]
            item[f"mean_{metric}"] = f"{float(np.mean(vals)):.5f}"
            item[f"ci95_{metric}"] = f"{ci95(vals):.5f}"
        rows.append(item)
    return rows


def build_pairwise(seed_rows: List[Dict[str, str]], reference: str = "impedance_token_policy") -> List[Dict[str, str]]:
    by_key = {(row["method"], row["split"], row["seed"]): row for row in seed_rows}
    rows: List[Dict[str, str]] = []
    methods = sorted({row["method"] for row in seed_rows if row["method"] != reference})
    for split in sorted({row["split"] for row in seed_rows}):
        for method in methods:
            success_diffs = []
            error_reductions = []
            safety_reductions = []
            chatter_reductions = []
            for seed in [str(s) for s in SEEDS]:
                ref = by_key.get((reference, split, seed))
                other = by_key.get((method, split, seed))
                if ref is None or other is None:
                    continue
                success_diffs.append(float(ref["success_rate"]) - float(other["success_rate"]))
                error_reductions.append(float(other["normalized_force_error"]) - float(ref["normalized_force_error"]))
                safety_reductions.append(float(other["safety_violation_rate"]) - float(ref["safety_violation_rate"]))
                chatter_reductions.append(float(other["chatter_rate"]) - float(ref["chatter_rate"]))
            if success_diffs:
                rows.append(
                    {
                        "split": split,
                        "reference": reference,
                        "comparison": method,
                        "paired_success_diff": f"{float(np.mean(success_diffs)):.5f}",
                        "ci95_success_diff": f"{ci95(success_diffs):.5f}",
                        "paired_force_error_reduction": f"{float(np.mean(error_reductions)):.5f}",
                        "paired_safety_reduction": f"{float(np.mean(safety_reductions)):.5f}",
                        "paired_chatter_reduction": f"{float(np.mean(chatter_reductions)):.5f}",
                        "reference_better_seeds": str(sum(1 for d in success_diffs if d > 0)),
                        "seeds": str(len(success_diffs)),
                    }
                )
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(summary: List[Dict[str, str]], split_order: Sequence[str], methods: Sequence[str], metric: str, title: str, path: Path, ylim: Tuple[float, float] | None = None) -> None:
    width = 0.095
    x = np.arange(len(split_order))
    plt.figure(figsize=(13, 5))
    for idx, method in enumerate(methods):
        vals = []
        errs = []
        for split in split_order:
            row = [r for r in summary if r["method"] == method and r["split"] == split][0]
            vals.append(float(row[f"mean_{metric}"]))
            errs.append(float(row[f"ci95_{metric}"]))
        plt.bar(x + (idx - len(methods) / 2) * width, vals, width, yerr=errs, label=method)
    plt.xticks(x, split_order, rotation=20, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_ablation(ablation_summary: List[Dict[str, str]], path: Path) -> None:
    rows = [row for row in ablation_summary if row["split"] == "combined_stress"]
    plt.figure(figsize=(10, 4.8))
    plt.bar([row["method"] for row in rows], [float(row["mean_success_rate"]) for row in rows], yerr=[float(row["ci95_success_rate"]) for row in rows], color="#705c53")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("success rate")
    plt.title("Paper 72 impedance-token ablations")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_stress(stress_summary: List[Dict[str, str]], path: Path) -> None:
    plt.figure(figsize=(9, 5))
    for method in sorted({row["method"] for row in stress_summary}):
        rows = sorted([row for row in stress_summary if row["method"] == method], key=lambda r: float(r["stress_level"]))
        x = [float(row["stress_level"]) for row in rows]
        y = [float(row["mean_success_rate"]) for row in rows]
        e = [float(row["ci95_success_rate"]) for row in rows]
        plt.errorbar(x, y, yerr=e, marker="o", label=method)
    plt.xlabel("stress level")
    plt.ylabel("success rate")
    plt.title("Paper 72 impedance control stress sweep")
    plt.ylim(0, 1.0)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def decide(summary: List[Dict[str, str]], pairwise: List[Dict[str, str]]) -> Tuple[str, str]:
    combined = [row for row in summary if row["split"] == "combined_stress"]
    proposed = [row for row in combined if row["method"] == "impedance_token_policy"][0]
    non_oracle = [row for row in combined if row["method"] not in {"impedance_token_policy", "oracle_impedance"}]
    best = max(non_oracle, key=lambda row: float(row["mean_success_rate"]))
    prop_success = float(proposed["mean_success_rate"])
    best_success = float(best["mean_success_rate"])
    prop_safety = float(proposed["mean_safety_violation_rate"])
    best_safety = float(best["mean_safety_violation_rate"])
    pair = [row for row in pairwise if row["split"] == "combined_stress" and row["comparison"] == best["method"]][0]
    paired = float(pair["paired_success_diff"])
    paired_ci = float(pair["ci95_success_diff"])
    if prop_success - best_success >= 0.045 and paired - paired_ci > 0.0 and prop_safety <= best_safety + 0.02:
        return (
            "STRONG_REVISE",
            f"impedance_token_policy beats strongest non-oracle baseline {best['method']} on combined_stress by "
            f"{prop_success - best_success:.3f} success with paired diff {paired:.3f}+/-{paired_ci:.3f}, "
            "but lacks real robot/public benchmark validation.",
        )
    return (
        "KILL_ARCHIVE",
        f"impedance_token_policy does not clear strongest non-oracle baseline {best['method']} decisively on combined_stress "
        f"(tokens={prop_success:.3f}, best_baseline={best_success:.3f}, paired diff={paired:.3f}+/-{paired_ci:.3f}).",
    )


def negative_cases(raw_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    candidates = [r for r in raw_rows if r["method"] == "impedance_token_policy" and r["split"] in {"combined_stress", "friction_slip_shift", "contact_transition"}]
    worst = sorted(candidates, key=lambda r: (int(r["success"]), -float(r["peak_overshoot"]), -float(r["safety_violation_rate"]), -float(r["normalized_force_error"])))[:12]
    rows: List[Dict[str, str]] = []
    for idx, row in enumerate(worst):
        lesson = "token selector did not adapt fast enough after contact shift"
        if float(row["safety_violation_rate"]) > 0.05:
            lesson = "safety penalty failed to prevent excessive penetration or force"
        elif float(row["slip_rate"]) > 0.35:
            lesson = "friction shift produced sliding chatter despite token adaptation"
        rows.append(
            {
                "case": str(idx),
                "split": row["split"],
                "seed": row["seed"],
                "episode": row["episode"],
                "success": row["success"],
                "normalized_force_error": row["normalized_force_error"],
                "peak_overshoot": row["peak_overshoot"],
                "safety_violation_rate": row["safety_violation_rate"],
                "slip_rate": row["slip_rate"],
                "lesson": lesson,
            }
        )
    return rows


def main() -> None:
    start_time = time.time()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    pack = generate_training_pack()
    write_csv(RESULTS / "training_impedance_examples.csv", pack.training_rows)
    write_csv(
        RESULTS / "training_summary.csv",
        [
            {
                "training_examples": str(TRAINING_EXAMPLES),
                "learned_gain_train_mae": f"{pack.train_mae:.5f}",
                "feature_dim": str(len(pack.scaler_x.mean_)),
                "target_dim": "4",
            }
        ],
    )

    model = make_model()
    raw_rows: List[Dict[str, str]] = []
    for split in SPLITS:
        for seed in SEEDS:
            for episode in range(EVAL_EPISODES):
                cfg = make_config(split, seed, episode)
                for method in METHODS:
                    raw_rows.append(simulate_episode(model, method, cfg, pack))
    write_csv(RESULTS / "impedance_raw.csv", raw_rows)
    write_csv(RESULTS / "impedance_rollouts.csv", raw_rows)
    seed_rows = build_seed_metrics(raw_rows)
    summary = build_summary(seed_rows)
    pairwise = build_pairwise(seed_rows)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "metrics.csv", summary)
    write_csv(RESULTS / "impedance_metrics.csv", summary)
    write_csv(RESULTS / "pairwise_stats.csv", pairwise)
    write_csv(RESULTS / "impedance_pairwise.csv", pairwise)

    combined = [s for s in SPLITS if s.name == "combined_stress"][0]
    ablation_raw: List[Dict[str, str]] = []
    for seed in SEEDS:
        for episode in range(ABLATION_EPISODES):
            cfg = make_config(combined, seed, 1000 + episode)
            for method in ABLATION_METHODS:
                row = simulate_episode(model, method, cfg, pack)
                row["method"] = method
                ablation_raw.append(row)
    write_csv(RESULTS / "impedance_ablation_raw.csv", ablation_raw)
    ablation_summary = build_summary(build_seed_metrics(ablation_raw))
    write_csv(RESULTS / "ablation_metrics.csv", ablation_summary)
    write_csv(RESULTS / "impedance_ablation.csv", ablation_summary)

    stress_raw: List[Dict[str, str]] = []
    for stress_level in np.linspace(0.0, 1.0, 6):
        for seed in SEEDS:
            for episode in range(STRESS_EPISODES):
                cfg = make_config(combined, seed, 2000 + episode, stress_level=float(stress_level))
                for method in STRESS_METHODS:
                    row = simulate_episode(model, method, cfg, pack)
                    row["split"] = "stress_sweep"
                    row["stress_level"] = f"{stress_level:.2f}"
                    stress_raw.append(row)
    write_csv(RESULTS / "stress_sweep_raw.csv", stress_raw)
    stress_summary: List[Dict[str, str]] = []
    stress_metrics = [
        "success",
        "normalized_force_error",
        "peak_overshoot",
        "safety_violation_rate",
        "slip_rate",
        "final_progress",
    ]
    for (method, stress_level), group in sorted(group_rows(stress_raw, ["method", "stress_level"]).items()):
        metric_seed_vals: Dict[str, List[float]] = {metric: [] for metric in stress_metrics}
        for seed in [str(s) for s in SEEDS]:
            rows = [r for r in group if r["seed"] == seed]
            if rows:
                for metric in stress_metrics:
                    metric_seed_vals[metric].append(float(np.mean([float(r[metric]) for r in rows])))
        item = {
            "method": method,
            "stress_level": stress_level,
            "seeds": str(len(metric_seed_vals["success"])),
            "episodes_per_seed": str(STRESS_EPISODES),
            "mean_success_rate": f"{float(np.mean(metric_seed_vals['success'])):.5f}",
            "ci95_success_rate": f"{ci95(metric_seed_vals['success']):.5f}",
        }
        for metric in stress_metrics[1:]:
            item[f"mean_{metric}"] = f"{float(np.mean(metric_seed_vals[metric])):.5f}"
            item[f"ci95_{metric}"] = f"{ci95(metric_seed_vals[metric]):.5f}"
        stress_summary.append(item)
    write_csv(RESULTS / "stress_sweep.csv", stress_summary)
    write_csv(FIGURES / "stress_curve_data.csv", stress_summary)
    write_csv(RESULTS / "negative_cases.csv", negative_cases(raw_rows))

    split_order = [s.name for s in SPLITS]
    plot_metric(summary, split_order, METHODS, "success_rate", "Paper 72 MuJoCo impedance-token success", FIGURES / "impedance_success_by_split.png", (0, 1.0))
    plot_metric(summary, split_order, METHODS, "normalized_force_error", "Paper 72 normalized force tracking error", FIGURES / "impedance_force_error_by_split.png")
    plot_metric(summary, split_order, METHODS, "safety_violation_rate", "Paper 72 safety violation rate", FIGURES / "impedance_safety_by_split.png", (0, 0.45))
    plot_ablation(ablation_summary, FIGURES / "impedance_ablation_success.png")
    plot_stress(stress_summary, FIGURES / "impedance_stress_sweep.png")

    decision, reason = decide(summary, pairwise)
    elapsed = time.time() - start_time
    combined_rows = [r for r in summary if r["split"] == "combined_stress"]
    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Paper 72 adaptive_impedance_tokens real MuJoCo rebuild\n")
        f.write(f"Terminal recommendation: {decision}\n")
        f.write(f"Reason: {reason}\n")
        f.write(f"Main eval rows: {len(raw_rows)}\n")
        f.write(f"Ablation rows: {len(ablation_raw)}\n")
        f.write(f"Stress rows: {len(stress_raw)}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write(f"Eval episodes per seed/split: {EVAL_EPISODES}\n")
        f.write(f"Runtime seconds: {elapsed:.2f}\n\n")
        f.write("Combined-stress summary:\n")
        for row in sorted(combined_rows, key=lambda r: -float(r["mean_success_rate"])):
            f.write(
                f"{row['method']} success={row['mean_success_rate']} ci95={row['ci95_success_rate']} "
                f"force_error={row['mean_normalized_force_error']} safety={row['mean_safety_violation_rate']} "
                f"chatter={row['mean_chatter_rate']}\n"
            )
    print(f"wrote Paper 72 MuJoCo impedance evidence to {RESULTS}")
    print(f"terminal recommendation: {decision}")
    print(reason)


if __name__ == "__main__":
    main()
