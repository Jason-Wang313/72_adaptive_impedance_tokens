from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


BASE_SEED = 235725184
DEFAULT_SEEDS = 7
DEFAULT_EPISODES = int(os.getenv("PAPER72_EVAL_EPISODES", "12"))
DEFAULT_ABLATION_EPISODES = int(os.getenv("PAPER72_ABLATION_EPISODES", "10"))
DEFAULT_STRESS_EPISODES = int(os.getenv("PAPER72_STRESS_EPISODES", "8"))
DEFAULT_TRAINING_EXAMPLES = int(os.getenv("PAPER72_TRAINING_EXAMPLES", "2600"))
STEPS = int(os.getenv("PAPER72_STEPS", "96"))
DT = 0.025
WALL_X = 0.0
Y_START = -0.43
Y_GOAL = 0.43
MAX_PENETRATION = 0.072

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_FIGURES = ROOT / "figures"

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

MAIN_METHODS = [
    "fixed_impedance",
    "gain_scheduled_impedance",
    "adaptive_impedance_control",
    "admittance_switching_control",
    "robust_mpc_impedance",
    "learned_gain_regressor",
    "random_forest_gain_regressor",
    "hist_gradient_gain_regressor",
    "ensemble_uncertainty_gain",
    "risk_averse_impedance",
    "conformal_safety_gain",
    "impedance_token_policy_v4",
    "impedance_token_policy_v5",
    "token_no_memory_ablation",
    "oracle_impedance",
]

ABLATION_METHODS = [
    "token_full_v5",
    "ablate_no_memory",
    "ablate_no_discrete_tokens",
    "ablate_no_force_update",
    "ablate_no_transition_planner",
    "ablate_no_safety_penalty",
    "ablate_no_slip_context",
    "ablate_no_tail_risk_objective",
    "ablate_no_calibration_guard",
    "ablate_no_phase_memory",
    "impedance_token_policy_v4",
    "learned_only_token_replacement",
]

STRESS_METHODS = [
    "gain_scheduled_impedance",
    "adaptive_impedance_control",
    "robust_mpc_impedance",
    "learned_gain_regressor",
    "random_forest_gain_regressor",
    "hist_gradient_gain_regressor",
    "ensemble_uncertainty_gain",
    "risk_averse_impedance",
    "conformal_safety_gain",
    "impedance_token_policy_v4",
    "impedance_token_policy_v5",
    "oracle_impedance",
]

HARD_SPLITS = {
    "stiffness_shift",
    "friction_slip_shift",
    "contact_transition",
    "target_force_jump",
    "actuator_saturation",
    "sensor_noise_burst",
    "stick_slip_cycle",
    "surface_discontinuity",
    "delayed_mode_switch",
    "combined_stress",
    "combined_extreme_stress",
}
COMBINED_SPLITS = {"combined_stress", "combined_extreme_stress"}


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
    burst_noise: float = 0.0
    stick_slip: float = 0.0
    surface_jump: float = 0.0
    delayed_shift: int = 0


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
    burst_start: int
    burst_end: int
    surface_jump: float
    stress_level: float | None = None


@dataclass
class MethodState:
    method: str
    k_est: float
    mu_est: float
    target_force_est: float
    token_scores: np.ndarray
    last_force: float
    force_error_integral: float
    chatter_crossings: int
    selected_token: int
    force_history: list[float]
    token_history: list[int]
    phase_memory: float
    cached_prediction_key: str
    cached_prediction_step: int
    cached_prediction: tuple[float, float, float, float, float]


@dataclass
class LearnedPack:
    scaler_x: StandardScaler
    scaler_y: StandardScaler
    ridge: Ridge
    random_forest: RandomForestRegressor
    hist_gradient: MultiOutputRegressor
    training_rows: list[dict[str, str]]
    ridge_mae: float
    forest_mae: float
    hist_mae: float
    conformal_margin: float


SPLITS = [
    SplitSpec("nominal_surface_tracking", 420.0, 1.00, 18.0, 0.18, 1.00, 12.0, 1.00, 0.28, 4.0, 0.20),
    SplitSpec("stiffness_shift", 360.0, 1.85, 19.0, 0.20, 1.00, 12.0, 1.00, 0.34, 3.6, 0.35),
    SplitSpec("friction_slip_shift", 430.0, 1.05, 16.0, 0.16, 2.35, 12.0, 1.00, 0.38, 3.5, 0.30, stick_slip=0.35),
    SplitSpec("contact_transition", 390.0, 1.40, 20.0, 0.22, 1.55, 11.0, 1.35, 0.42, 3.25, 0.65),
    SplitSpec("target_force_jump", 405.0, 1.20, 17.0, 0.21, 1.35, 9.5, 1.85, 0.44, 3.35, 0.55),
    SplitSpec("actuator_saturation", 380.0, 1.55, 18.0, 0.19, 1.45, 12.0, 1.45, 0.46, 2.45, 0.70),
    SplitSpec("sensor_noise_burst", 410.0, 1.25, 17.0, 0.18, 1.30, 12.5, 1.30, 0.36, 3.20, 0.55, burst_noise=1.25),
    SplitSpec("stick_slip_cycle", 420.0, 1.10, 15.0, 0.14, 2.70, 11.5, 1.20, 0.46, 3.10, 0.62, stick_slip=0.70),
    SplitSpec("surface_discontinuity", 370.0, 1.70, 18.0, 0.22, 1.60, 11.0, 1.50, 0.48, 2.90, 0.72, surface_jump=0.014),
    SplitSpec("delayed_mode_switch", 350.0, 2.05, 17.0, 0.19, 2.10, 10.8, 1.55, 0.52, 2.95, 0.85, delayed_shift=16),
    SplitSpec("combined_stress", 340.0, 2.25, 17.0, 0.18, 2.70, 10.5, 1.60, 0.62, 2.85, 0.85, burst_noise=0.75, stick_slip=0.45, surface_jump=0.010),
    SplitSpec("combined_extreme_stress", 315.0, 2.75, 16.0, 0.16, 3.05, 10.0, 1.85, 0.82, 2.35, 0.98, burst_noise=1.10, stick_slip=0.80, surface_jump=0.018, delayed_shift=10),
]

TOKEN_TABLE = np.array(
    [
        [60.0, 9.0, 0.82, 7.0, 1.10],
        [90.0, 11.0, 1.00, 8.5, 1.00],
        [135.0, 15.0, 1.05, 9.0, 0.90],
        [115.0, 22.0, 0.92, 6.5, 1.35],
        [160.0, 24.0, 0.78, 5.5, 1.70],
        [100.0, 18.0, 1.25, 7.5, 1.15],
        [72.0, 30.0, 0.68, 4.7, 2.10],
        [150.0, 12.0, 1.15, 9.3, 0.75],
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
    if stress_level is None:
        stiffness = split.stiffness * rng.normal(1.0, 0.08)
        stiffness_after = split.stiffness * split.stiffness_shift * rng.normal(1.0, 0.06)
        friction = split.friction * rng.normal(1.0, 0.08)
        friction_after = split.friction * split.friction_shift * rng.normal(1.0, 0.08)
        target = split.target_force * rng.normal(1.0, 0.05)
        target_after = split.target_force * split.target_force_shift * rng.normal(1.0, 0.05)
        noise = split.force_noise
        actuator_limit = split.actuator_limit
        surface_jump = split.surface_jump
    else:
        stress = float(stress_level)
        stiffness = (360.0 + 60.0 * rng.normal()) * (1.0 + 0.25 * stress)
        stiffness_after = stiffness * (1.0 + 1.85 * stress)
        friction = 0.14 + 0.06 * rng.random()
        friction_after = friction * (1.0 + 2.25 * stress)
        target = 10.0 + 2.0 * rng.random()
        target_after = target * (1.0 + 0.85 * stress)
        noise = 0.24 + 0.78 * stress
        actuator_limit = 4.0 - 1.55 * stress
        surface_jump = 0.018 * stress
    shift_step = int(rng.integers(42, 62)) + split.delayed_shift
    target_shift_step = int(rng.integers(58, 76)) + split.delayed_shift // 2
    burst_start = int(rng.integers(36, 50))
    burst_end = burst_start + int(rng.integers(14, 24))
    return EpisodeConfig(
        split=split,
        seed=seed,
        episode=episode,
        stiffness=max(160.0, stiffness),
        stiffness_after=max(180.0, stiffness_after),
        damping=max(8.0, split.damping * rng.normal(1.0, 0.07)),
        friction=max(0.05, friction),
        friction_after=max(0.06, friction_after),
        target_force=max(6.0, target),
        target_force_after=max(6.0, target_after),
        force_noise=max(0.02, noise),
        actuator_limit=max(1.5, actuator_limit),
        shift_step=min(STEPS - 12, shift_step),
        target_shift_step=min(STEPS - 10, target_shift_step),
        burst_start=burst_start,
        burst_end=burst_end,
        surface_jump=surface_jump,
        stress_level=stress_level,
    )


def active_surface(cfg: EpisodeConfig, step: int) -> tuple[float, float, float, float]:
    stiffness = cfg.stiffness_after if step >= cfg.shift_step else cfg.stiffness
    friction = cfg.friction_after if step >= cfg.shift_step else cfg.friction
    target = cfg.target_force_after if step >= cfg.target_shift_step else cfg.target_force
    jump = cfg.surface_jump if step >= cfg.shift_step else 0.0
    return stiffness, friction, target, jump


def init_state(method: str, cfg: EpisodeConfig) -> MethodState:
    scores = np.zeros(len(TOKEN_TABLE), dtype=float)
    scores[1] = 0.4
    return MethodState(
        method=method,
        k_est=420.0,
        mu_est=0.20,
        target_force_est=cfg.target_force,
        token_scores=scores,
        last_force=0.0,
        force_error_integral=0.0,
        chatter_crossings=0,
        selected_token=1,
        force_history=[],
        token_history=[],
        phase_memory=0.0,
        cached_prediction_key="",
        cached_prediction_step=-999,
        cached_prediction=(90.0, 12.0, 7.0, 1.0, 0.0),
    )


def surface_force(
    x: float,
    vx: float,
    vy: float,
    stiffness: float,
    damping: float,
    friction: float,
    stick_slip: float,
    surface_jump: float,
    step: int,
) -> tuple[float, float, float]:
    penetration = max(0.0, x - (WALL_X + surface_jump))
    if penetration <= 0.0:
        return 0.0, 0.0, 0.0
    slip_wave = 1.0 + stick_slip * (0.5 + 0.5 * math.sin(0.31 * step + 7.0 * penetration))
    normal = stiffness * penetration + damping * max(0.0, vx)
    tangent = -friction * slip_wave * normal * math.tanh(vy / 0.050)
    return normal, tangent, penetration


def train_feature(
    force: float,
    penetration: float,
    vx: float,
    vy: float,
    target_force: float,
    k_est: float,
    mu_est: float,
    phase: float,
    force_error: float,
    slip_rate: float,
    chatter: float,
    safety_margin: float,
) -> np.ndarray:
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
            slip_rate,
            chatter,
        ],
        dtype=float,
    )


def analytic_gains(
    stiffness: float,
    friction: float,
    target: float,
    penetration: float,
    force_error: float,
    slip_rate: float,
    safety: float,
    phase: float,
) -> np.ndarray:
    desired_pen = target / max(210.0, stiffness)
    normal_kp = np.clip(65.0 + 0.15 * stiffness + 2.2 * abs(force_error) - 980.0 * safety, 45.0, 205.0)
    normal_kd = np.clip(8.0 + 0.030 * stiffness + 8.0 * friction + 26.0 * safety + 3.0 * phase, 6.0, 36.0)
    y_gain = np.clip(9.2 - 4.2 * friction - 1.6 * slip_rate - 35.0 * safety, 3.0, 11.0)
    target_scale = np.clip(desired_pen / max(1e-4, target / 420.0), 0.42, 2.00)
    return np.array([normal_kp, normal_kd, y_gain, target_scale], dtype=float)


def generate_training_pack(train_scenes: int) -> LearnedPack:
    rng = np.random.default_rng(BASE_SEED + 515)
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    csv_rows: list[dict[str, str]] = []
    for idx in range(train_scenes):
        stiffness = rng.uniform(190.0, 960.0)
        friction = rng.uniform(0.06, 0.86)
        target = rng.uniform(6.5, 22.0)
        penetration = rng.uniform(0.0, min(MAX_PENETRATION, target / stiffness * rng.uniform(0.35, 1.90)))
        vx = rng.normal(0.0, 0.24)
        vy = rng.normal(0.18, 0.36)
        normal = stiffness * penetration + rng.normal(0.0, 0.35)
        force_error = target - normal
        k_est = np.clip(normal / max(0.006, penetration), 170.0, 950.0) if penetration > 0.004 else 420.0
        mu_est = np.clip(friction + rng.normal(0.0, 0.08), 0.05, 0.85)
        phase = 1.0 if penetration > 0.005 else 0.0
        slip_rate = min(1.0, abs(vy) / 0.55)
        chatter = rng.uniform(0.0, 0.20) if abs(force_error) > 2.0 else rng.uniform(0.0, 0.05)
        safety = max(0.0, penetration - 0.045)
        feat = train_feature(normal, penetration, vx, vy, target, k_est, mu_est, phase, force_error, slip_rate, chatter, safety)
        target_gains = analytic_gains(stiffness, friction, target, penetration, force_error, slip_rate, safety, phase)
        x_rows.append(feat)
        y_rows.append(target_gains)
        csv_rows.append(
            {
                "example": str(idx),
                "stiffness": f"{stiffness:.4f}",
                "friction": f"{friction:.4f}",
                "target_force": f"{target:.4f}",
                "penetration": f"{penetration:.5f}",
                "force": f"{normal:.4f}",
                "normal_kp": f"{target_gains[0]:.4f}",
                "normal_kd": f"{target_gains[1]:.4f}",
                "y_gain": f"{target_gains[2]:.4f}",
                "target_scale": f"{target_gains[3]:.4f}",
            }
        )
    x = np.vstack(x_rows)
    y = np.vstack(y_rows)
    scaler_x = StandardScaler().fit(x)
    scaler_y = StandardScaler().fit(y)
    x_scaled = scaler_x.transform(x)
    y_scaled = scaler_y.transform(y)
    ridge = Ridge(alpha=0.08).fit(x_scaled, y_scaled)
    forest = RandomForestRegressor(n_estimators=28, max_depth=8, min_samples_leaf=5, random_state=BASE_SEED, n_jobs=1).fit(x_scaled, y)
    hist = MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=45, max_leaf_nodes=12, learning_rate=0.07, l2_regularization=0.05, random_state=BASE_SEED)
    ).fit(x_scaled, y)

    ridge_pred = scaler_y.inverse_transform(ridge.predict(x_scaled))
    forest_pred = forest.predict(x_scaled)
    hist_pred = hist.predict(x_scaled)
    ridge_mae = float(np.mean(np.abs(ridge_pred - y)))
    forest_mae = float(np.mean(np.abs(forest_pred - y)))
    hist_mae = float(np.mean(np.abs(hist_pred - y)))
    residual = np.mean(np.abs(np.vstack([ridge_pred, forest_pred, hist_pred]) - np.vstack([y, y, y])), axis=1)
    conformal_margin = float(np.quantile(residual, 0.90))
    return LearnedPack(
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        ridge=ridge,
        random_forest=forest,
        hist_gradient=hist,
        training_rows=csv_rows,
        ridge_mae=ridge_mae,
        forest_mae=forest_mae,
        hist_mae=hist_mae,
        conformal_margin=conformal_margin,
    )


def clipped_gains(values: Sequence[float]) -> tuple[float, float, float, float]:
    return (
        float(np.clip(values[0], 35.0, 215.0)),
        float(np.clip(values[1], 5.0, 38.0)),
        float(np.clip(values[2], 2.6, 11.5)),
        float(np.clip(values[3], 0.38, 2.15)),
    )


def learned_prediction(pack: LearnedPack, feat: np.ndarray, which: str) -> tuple[float, float, float, float, float]:
    x_scaled = pack.scaler_x.transform(feat.reshape(1, -1))
    ridge_pred = pack.scaler_y.inverse_transform(pack.ridge.predict(x_scaled))[0]
    forest_pred = pack.random_forest.predict(x_scaled)[0]
    hist_pred = pack.hist_gradient.predict(x_scaled)[0]
    if which == "ridge":
        pred = ridge_pred
    elif which == "forest":
        pred = forest_pred
    elif which == "hist":
        pred = hist_pred
    else:
        pred = np.mean([ridge_pred, forest_pred, hist_pred], axis=0)
    disagreement = float(np.mean(np.std(np.vstack([ridge_pred, forest_pred, hist_pred]), axis=0)))
    return (*clipped_gains(pred), disagreement)


def cached_learned_prediction(
    state: MethodState,
    pack: LearnedPack,
    feat: np.ndarray,
    which: str,
    step: int,
    interval: int = 8,
) -> tuple[float, float, float, float, float]:
    key = which
    if state.cached_prediction_key == key and step - state.cached_prediction_step < interval:
        return state.cached_prediction
    pred = learned_prediction(pack, feat, which)
    state.cached_prediction_key = key
    state.cached_prediction_step = step
    state.cached_prediction = pred
    return pred


def choose_token_v4(state: MethodState, force: float, penetration: float, slip: float, target_force: float, method: str) -> int:
    scores = state.token_scores.copy()
    force_error = target_force - force
    if penetration < 0.006:
        scores += np.array([0.2, 0.5, 0.1, 0.8, 0.2, 0.0, 0.1, 0.0])
    if abs(force_error) > 3.0:
        scores += np.array([0.1, 0.2, 0.5, 0.1, -0.2, 0.6, 0.1, 0.0])
    if force > 1.45 * target_force or penetration > 0.045:
        scores += np.array([0.3, 0.1, -0.1, 0.7, 1.2, -0.2, 0.5, -0.1])
    if abs(slip) > 0.34:
        scores += np.array([0.0, 0.1, 0.2, 0.9, 0.8, -0.1, 0.6, 0.2])
    if method == "ablate_no_discrete_tokens":
        return 1
    return int(np.argmax(scores))


def choose_token_v5(
    state: MethodState,
    force: float,
    penetration: float,
    slip: float,
    target_force: float,
    phase: float,
    uncertainty: float,
    method: str,
) -> int:
    if method == "ablate_no_discrete_tokens":
        return 1
    scores = 0.86 * state.token_scores.copy()
    force_error = target_force - force
    high_risk = penetration > 0.047 or force > 1.45 * target_force or uncertainty > 8.0
    if penetration < 0.006:
        scores += np.array([0.1, 0.4, 0.1, 0.8, 0.1, 0.1, 0.2, 0.0])
    if abs(force_error) > 2.5:
        scores += np.array([0.0, 0.2, 0.4, 0.1, -0.1, 0.5, 0.1, 0.3])
    if high_risk:
        scores += np.array([0.3, 0.0, -0.1, 0.6, 1.3, -0.2, 1.0, -0.1])
    if abs(slip) > 0.28 and method != "ablate_no_slip_context":
        scores += np.array([0.0, 0.1, 0.2, 1.0, 0.7, -0.1, 0.5, 0.4])
    if phase > 0.5 and method != "ablate_no_phase_memory":
        scores += np.array([0.0, 0.1, 0.3, 0.0, 0.3, 0.2, 0.5, 0.2])
    if method == "ablate_no_tail_risk_objective":
        scores[4] -= 0.8
        scores[6] -= 0.8
    if method == "ablate_no_calibration_guard":
        scores[7] += 0.4
        scores[4] -= 0.2
    return int(np.argmax(scores))


def controller_gains(
    method: str,
    state: MethodState,
    pack: LearnedPack,
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
    slip_rate: float,
    chatter_rate: float,
) -> tuple[float, float, float, float, int]:
    contact = penetration > 0.002
    k_obs = np.clip(force_obs / max(0.004, penetration), 170.0, 980.0) if contact else state.k_est
    if contact and method not in {"ablate_no_force_update"}:
        state.k_est = 0.90 * state.k_est + 0.10 * k_obs
        slip_mu = min(0.9, abs(vy) / max(0.02, force_obs + 1.0))
        state.mu_est = 0.94 * state.mu_est + 0.06 * max(0.05, min(0.85, true_friction + 0.08 * slip_mu))
    state.force_error_integral = np.clip(state.force_error_integral + (target_force - force_obs) * DT, -25.0, 25.0)
    force_error = target_force - force_obs
    state.phase_memory = 0.94 * state.phase_memory + 0.06 * (1.0 if contact else 0.0)
    phase = state.phase_memory
    safety_margin = max(0.0, penetration - 0.045)
    feat = train_feature(force_obs, penetration, vx, vy, target_force, state.k_est, state.mu_est, phase, force_error, slip_rate, chatter_rate, safety_margin)
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
        desired_pen = 0.86 * target_force / max(260.0, state.k_est)
        return float(np.clip(kp, 65.0, 145.0)), float(np.clip(kd, 15.0, 36.0)), y_gain, desired_pen, token_idx
    if method == "learned_gain_regressor":
        kp, kd, y_gain, scale, _ = cached_learned_prediction(state, pack, feat, "ridge", step)
        return kp, kd, y_gain, scale * target_force / 420.0, token_idx
    if method == "random_forest_gain_regressor":
        kp, kd, y_gain, scale, _ = cached_learned_prediction(state, pack, feat, "forest", step)
        return kp, kd, y_gain, scale * target_force / 420.0, token_idx
    if method == "hist_gradient_gain_regressor":
        kp, kd, y_gain, scale, _ = cached_learned_prediction(state, pack, feat, "hist", step)
        return kp, kd, y_gain, scale * target_force / 420.0, token_idx
    if method == "ensemble_uncertainty_gain":
        kp, kd, y_gain, scale, disagreement = cached_learned_prediction(state, pack, feat, "ensemble", step)
        safety_scale = 1.0 - min(0.18, disagreement / 120.0) - 0.8 * safety_margin
        return kp, kd + 0.4 * disagreement, max(2.8, y_gain - 0.08 * disagreement), safety_scale * scale * target_force / 420.0, token_idx
    if method == "risk_averse_impedance":
        kp = 78.0 + 0.10 * min(800.0, state.k_est) + 0.8 * abs(force_error)
        kd = 22.0 + 0.014 * state.k_est + 10.0 * safety_margin
        y_gain = 4.2 - 1.5 * min(0.8, state.mu_est)
        desired_pen = 0.76 * target_force / max(270.0, state.k_est)
        return float(np.clip(kp, 62.0, 160.0)), float(np.clip(kd, 14.0, 38.0)), float(np.clip(y_gain, 2.8, 7.0)), desired_pen, token_idx
    if method == "conformal_safety_gain":
        kp, kd, y_gain, scale, disagreement = learned_prediction(pack, feat, "ensemble")
        margin = min(0.30, (pack.conformal_margin + disagreement) / 150.0)
        desired_pen = (1.0 - margin) * scale * target_force / 420.0
        return kp, min(38.0, kd + 2.0 * margin), max(2.8, y_gain - 4.0 * margin), desired_pen, token_idx
    if method in {"impedance_token_policy_v4"}:
        token_idx = choose_token_v4(state, force_obs, penetration, vy, target_force, method)
        token = TOKEN_TABLE[token_idx]
        reward = -abs(force_error) / max(1.0, target_force) - 1.2 * max(0.0, penetration - 0.048) - 0.15 * abs(vy)
        state.token_scores[token_idx] = 0.92 * state.token_scores[token_idx] + 0.08 * reward
        kp = token[0] + 0.05 * state.k_est
        kd = token[1] + 0.010 * state.k_est
        y_gain = token[3] - 2.4 * min(0.8, state.mu_est)
        desired_pen = token[2] * target_force / max(210.0, state.k_est)
        return float(np.clip(kp, 45.0, 205.0)), float(np.clip(kd, 6.0, 36.0)), float(np.clip(y_gain, 2.8, 10.5)), desired_pen, token_idx
    if method in {
        "impedance_token_policy_v5",
        "token_no_memory_ablation",
        "token_full_v5",
        "ablate_no_memory",
        "ablate_no_discrete_tokens",
        "ablate_no_force_update",
        "ablate_no_transition_planner",
        "ablate_no_safety_penalty",
        "ablate_no_slip_context",
        "ablate_no_tail_risk_objective",
        "ablate_no_calibration_guard",
        "ablate_no_phase_memory",
    }:
        ablation_method = {
            "token_no_memory_ablation": "ablate_no_memory",
            "token_full_v5": "impedance_token_policy_v5",
        }.get(method, method)
        if ablation_method == "ablate_no_memory":
            state.token_scores[:] = 0.0
        _, _, _, _, disagreement = cached_learned_prediction(state, pack, feat, "ensemble", step, interval=10)
        token_idx = choose_token_v5(state, force_obs, penetration, vy, target_force, phase, disagreement, ablation_method)
        token = TOKEN_TABLE[token_idx]
        if ablation_method != "ablate_no_memory":
            tail_penalty = 0.0 if ablation_method == "ablate_no_tail_risk_objective" else 1.8 * max(0.0, penetration - 0.046)
            slip_penalty = 0.0 if ablation_method == "ablate_no_slip_context" else 0.18 * abs(vy)
            reward = -abs(force_error) / max(1.0, target_force) - tail_penalty - slip_penalty - 0.04 * disagreement
            state.token_scores[token_idx] = 0.90 * state.token_scores[token_idx] + 0.10 * reward
        kp = token[0] + 0.045 * state.k_est + 0.7 * abs(force_error)
        kd = token[1] + 0.012 * state.k_est + 2.0 * safety_margin
        y_gain = token[3] - 2.0 * min(0.8, state.mu_est)
        if ablation_method == "ablate_no_transition_planner" and not contact:
            y_gain += 1.8
        if ablation_method == "ablate_no_safety_penalty":
            safety_scale = token[2]
        else:
            safety_scale = min(token[2], 0.80) if penetration > 0.050 or force_obs > 1.55 * target_force else token[2]
        if ablation_method != "ablate_no_calibration_guard":
            safety_scale -= min(0.12, disagreement / 140.0)
        desired_pen = safety_scale * target_force / max(205.0, state.k_est)
        return float(np.clip(kp, 45.0, 210.0)), float(np.clip(kd, 6.0, 38.0)), float(np.clip(y_gain, 2.8, 10.8)), desired_pen, token_idx
    if method == "learned_only_token_replacement":
        kp, kd, y_gain, scale, disagreement = cached_learned_prediction(state, pack, feat, "ensemble", step)
        return kp, kd, max(2.8, y_gain - 0.05 * disagreement), scale * target_force / 420.0, token_idx
    if method == "oracle_impedance":
        kp, kd, y_gain, scale = analytic_gains(true_stiffness, true_friction, target_force, penetration, force_error, slip_rate, safety_margin, phase)
        desired_pen = 1.04 * scale * target_force / 420.0
        return kp, kd, y_gain, desired_pen, token_idx
    raise ValueError(f"unknown method {method}")


def simulate_episode(model: mujoco.MjModel, method: str, cfg: EpisodeConfig, pack: LearnedPack) -> dict[str, str]:
    rng = np.random.default_rng(BASE_SEED + 1019 * cfg.seed + 337 * cfg.episode + sum(ord(c) for c in method))
    data = mujoco.MjData(model)
    data.qpos[:2] = np.array([-0.115 + rng.normal(0.0, 0.006), Y_START + rng.normal(0.0, 0.008)])
    data.qvel[:2] = 0.0
    mujoco.mj_forward(model, data)
    state = init_state(method, cfg)

    abs_force_errors: list[float] = []
    post_contact_errors: list[float] = []
    samples: list[str] = []
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
        stiffness, friction, target_force, surface_jump = active_surface(cfg, step)
        normal_force, tangential_force, penetration = surface_force(
            x, vx, vy, stiffness, cfg.damping, friction, cfg.split.stick_slip, surface_jump, step
        )
        burst = cfg.split.burst_noise if cfg.burst_start <= step <= cfg.burst_end else 0.0
        observed_force = max(0.0, normal_force + rng.normal(0.0, cfg.force_noise + burst))
        force_error = target_force - observed_force

        contact = penetration > 0.002
        if contact:
            contact_steps += 1
        if penetration > 0.050 or observed_force > 1.70 * target_force:
            safety_steps += 1
        if contact and abs(vy) > 0.34:
            slip_steps += 1
        if abs(force_error) > 0.0:
            sign = 1 if force_error > 0 else -1
            if last_error_sign != 0 and sign != last_error_sign and abs(force_error) > 1.0 and contact:
                state.chatter_crossings += 1
                chatter_steps += 1
            last_error_sign = sign

        slip_rate_so_far = slip_steps / max(1, contact_steps)
        chatter_rate_so_far = chatter_steps / max(1, contact_steps)
        kp, kd, y_gain, desired_pen, token_idx = controller_gains(
            method,
            state,
            pack,
            step,
            x,
            y,
            vx,
            vy,
            observed_force,
            penetration,
            target_force,
            stiffness,
            friction,
            slip_rate_so_far,
            chatter_rate_so_far,
        )
        state.selected_token = token_idx
        state.token_history.append(token_idx)
        state.force_history.append(observed_force)
        abs_force_errors.append(abs(force_error))
        if step >= cfg.target_shift_step:
            post_contact_errors.append(abs(force_error))
        overshoot = max(overshoot, observed_force / max(1.0, target_force))
        max_penetration = max(max_penetration, penetration)
        if settled_step is None and step > cfg.target_shift_step + 4 and abs(force_error) / max(1.0, target_force) < 0.20:
            settled_step = step

        desired_x = WALL_X + surface_jump + float(np.clip(desired_pen, 0.004, MAX_PENETRATION))
        progress = min(1.0, step / (STEPS - 20))
        desired_y = Y_START + (Y_GOAL - Y_START) * progress
        if penetration < 0.004 and step < 26:
            desired_y = y
        normal_ctrl = kp * (desired_x - x) - kd * vx + 0.16 * force_error
        tangent_ctrl = y_gain * (desired_y - y) - 5.0 * vy + 0.035 * tangential_force
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
        and final_progress > 0.74
        and normalized_error < 0.23
        and overshoot < 2.05
        and safety_rate < 0.10
        and slip_rate < 0.60
        and chatter_rate < 0.22
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


def group_rows(rows: Iterable[dict[str, str]], fields: Sequence[str]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in fields), []).append(row)
    return grouped


def mean_metric(rows: Sequence[dict[str, str]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_seed_metrics(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    metrics = [
        "success",
        "mean_abs_force_error",
        "post_shift_force_error",
        "normalized_force_error",
        "peak_overshoot",
        "safety_violation_rate",
        "slip_rate",
        "chatter_rate",
        "settling_latency",
        "energy",
        "final_progress",
        "token_switches",
    ]
    for (method, split, seed), group in sorted(group_rows(rows, ["method", "split", "seed"]).items()):
        item = {"method": method, "split": split, "seed": seed, "episodes": str(len(group))}
        for metric in metrics:
            label = "success_rate" if metric == "success" else f"mean_{metric}"
            item[label] = f"{mean_metric(group, metric):.5f}"
        out.append(item)
    return out


def build_summary(seed_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    metrics = [name for name in seed_rows[0].keys() if name not in {"method", "split", "seed", "episodes"}]
    rows: list[dict[str, str]] = []
    for (method, split), group in sorted(group_rows(seed_rows, ["method", "split"]).items()):
        item: dict[str, str] = {"method": method, "split": split, "seeds": str(len(group)), "episodes_per_seed": group[0]["episodes"]}
        for metric in metrics:
            vals = [float(row[metric]) for row in group]
            item[f"mean_{metric}"] = f"{float(np.mean(vals)):.5f}"
            item[f"ci95_{metric}"] = f"{ci95(vals):.5f}"
        rows.append(item)
    return rows


def build_pairwise(seed_rows: list[dict[str, str]], reference: str = "impedance_token_policy_v5") -> list[dict[str, str]]:
    by_key = {(row["method"], row["split"], row["seed"]): row for row in seed_rows}
    rows: list[dict[str, str]] = []
    methods = sorted({row["method"] for row in seed_rows if row["method"] != reference})
    seeds = sorted({row["seed"] for row in seed_rows}, key=lambda x: int(float(x)))
    for split in sorted({row["split"] for row in seed_rows}):
        for method in methods:
            success_diffs = []
            error_reductions = []
            safety_reductions = []
            slip_reductions = []
            chatter_reductions = []
            for seed in seeds:
                ref = by_key.get((reference, split, seed))
                other = by_key.get((method, split, seed))
                if ref is None or other is None:
                    continue
                success_diffs.append(float(ref["success_rate"]) - float(other["success_rate"]))
                error_reductions.append(float(other["mean_normalized_force_error"]) - float(ref["mean_normalized_force_error"]))
                safety_reductions.append(float(other["mean_safety_violation_rate"]) - float(ref["mean_safety_violation_rate"]))
                slip_reductions.append(float(other["mean_slip_rate"]) - float(ref["mean_slip_rate"]))
                chatter_reductions.append(float(other["mean_chatter_rate"]) - float(ref["mean_chatter_rate"]))
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
                        "paired_slip_reduction": f"{float(np.mean(slip_reductions)):.5f}",
                        "paired_chatter_reduction": f"{float(np.mean(chatter_reductions)):.5f}",
                        "reference_better_seeds": str(sum(1 for d in success_diffs if d > 0)),
                        "seeds": str(len(success_diffs)),
                    }
                )
    return rows


def build_aggregate(seed_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups = {
        "all_splits": {row["split"] for row in seed_rows},
        "hard_splits": HARD_SPLITS,
        "combined_and_extreme": COMBINED_SPLITS,
    }
    metrics = [
        "success_rate",
        "mean_normalized_force_error",
        "mean_peak_overshoot",
        "mean_safety_violation_rate",
        "mean_slip_rate",
        "mean_chatter_rate",
        "mean_settling_latency",
        "mean_energy",
    ]
    rows: list[dict[str, str]] = []
    for group_name, split_set in groups.items():
        for method in sorted({row["method"] for row in seed_rows}):
            selected = [row for row in seed_rows if row["method"] == method and row["split"] in split_set]
            if not selected:
                continue
            item = {"group": group_name, "method": method, "seed_split_rows": str(len(selected))}
            for metric in metrics:
                vals = [float(row[metric]) for row in selected]
                short = metric.replace("mean_", "")
                item[short] = f"{float(np.mean(vals)):.5f}"
                item[f"ci95_{short}"] = f"{ci95(vals):.5f}"
            rows.append(item)
    return rows


def build_group_pairwise(seed_rows: list[dict[str, str]], reference: str = "impedance_token_policy_v5") -> list[dict[str, str]]:
    groups = {
        "hard_splits": HARD_SPLITS,
        "combined_and_extreme": COMBINED_SPLITS,
    }
    rows: list[dict[str, str]] = []
    methods = sorted({row["method"] for row in seed_rows if row["method"] != reference})
    for group_name, split_set in groups.items():
        for method in methods:
            diffs = []
            for seed in sorted({row["seed"] for row in seed_rows}, key=lambda x: int(float(x))):
                ref_vals = [float(row["success_rate"]) for row in seed_rows if row["method"] == reference and row["split"] in split_set and row["seed"] == seed]
                other_vals = [float(row["success_rate"]) for row in seed_rows if row["method"] == method and row["split"] in split_set and row["seed"] == seed]
                if ref_vals and other_vals:
                    diffs.append(float(np.mean(ref_vals)) - float(np.mean(other_vals)))
            if diffs:
                rows.append(
                    {
                        "group": group_name,
                        "reference": reference,
                        "comparison": method,
                        "paired_success_diff": f"{float(np.mean(diffs)):.5f}",
                        "ci95_success_diff": f"{ci95(diffs):.5f}",
                        "reference_better_seeds": str(sum(1 for d in diffs if d > 0)),
                        "seeds": str(len(diffs)),
                    }
                )
    return rows


def build_fixed_risk(raw_rows: list[dict[str, str]], budgets: Sequence[float] = (0.05, 0.10, 0.20)) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    hard_raw = [row for row in raw_rows if row["split"] in HARD_SPLITS]
    for budget in budgets:
        for method in sorted({row["method"] for row in hard_raw}):
            selected = [row for row in hard_raw if row["method"] == method]
            safe = [
                float(row["success"])
                if float(row["safety_violation_rate"]) <= budget and float(row["slip_rate"]) <= 0.46 and float(row["chatter_rate"]) <= 0.18
                else 0.0
                for row in selected
            ]
            rows.append(
                {
                    "budget": f"{budget:.2f}",
                    "method": method,
                    "episodes": str(len(selected)),
                    "success_at_budget": f"{float(np.mean(safe)):.5f}",
                    "mean_safety_violation_rate": f"{float(np.mean([float(row['safety_violation_rate']) for row in selected])):.5f}",
                    "mean_slip_rate": f"{float(np.mean([float(row['slip_rate']) for row in selected])):.5f}",
                    "mean_chatter_rate": f"{float(np.mean([float(row['chatter_rate']) for row in selected])):.5f}",
                }
            )
    return rows


def build_ablation_aggregate(ablation_summary: list[dict[str, str]]) -> list[dict[str, str]]:
    def metric(row: dict[str, str], name: str) -> float:
        for key in (name, f"mean_{name}", f"mean_mean_{name}"):
            if key in row:
                return float(row[key])
        raise KeyError(name)

    rows: list[dict[str, str]] = []
    for method in sorted({row["method"] for row in ablation_summary}):
        selected = [row for row in ablation_summary if row["method"] == method]
        rows.append(
            {
                "method": method,
                "split_rows": str(len(selected)),
                "success": f"{float(np.mean([metric(row, 'success_rate') for row in selected])):.5f}",
                "normalized_force_error": f"{float(np.mean([metric(row, 'normalized_force_error') for row in selected])):.5f}",
                "safety_violation_rate": f"{float(np.mean([metric(row, 'safety_violation_rate') for row in selected])):.5f}",
                "slip_rate": f"{float(np.mean([metric(row, 'slip_rate') for row in selected])):.5f}",
                "chatter_rate": f"{float(np.mean([metric(row, 'chatter_rate') for row in selected])):.5f}",
            }
        )
    return rows


def build_stress_summary(stress_raw: list[dict[str, str]], stress_episodes: int) -> list[dict[str, str]]:
    stress_summary: list[dict[str, str]] = []
    metrics = ["success", "normalized_force_error", "peak_overshoot", "safety_violation_rate", "slip_rate", "chatter_rate", "final_progress"]
    for (method, split, stress_level), group in sorted(group_rows(stress_raw, ["method", "split", "stress_level"]).items()):
        item = {
            "method": method,
            "split": split,
            "stress_level": stress_level,
            "seeds": str(len({row["seed"] for row in group})),
            "episodes_per_seed": str(stress_episodes),
        }
        for metric in metrics:
            seed_vals = []
            for seed in sorted({row["seed"] for row in group}, key=lambda x: int(float(x))):
                rows = [row for row in group if row["seed"] == seed]
                seed_vals.append(float(np.mean([float(row[metric]) for row in rows])))
            out_name = "success_rate" if metric == "success" else metric
            item[f"mean_{out_name}"] = f"{float(np.mean(seed_vals)):.5f}"
            item[f"ci95_{out_name}"] = f"{ci95(seed_vals):.5f}"
        stress_summary.append(item)
    return stress_summary


def negative_cases(raw_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    candidates = [r for r in raw_rows if r["method"] == "impedance_token_policy_v5" and r["split"] in HARD_SPLITS]
    worst = sorted(
        candidates,
        key=lambda r: (
            int(r["success"]),
            -float(r["peak_overshoot"]),
            -float(r["safety_violation_rate"]),
            -float(r["slip_rate"]),
            -float(r["normalized_force_error"]),
        ),
    )[:12]
    rows: list[dict[str, str]] = []
    for idx, row in enumerate(worst):
        lesson = "token selector did not adapt fast enough after contact shift"
        if float(row["safety_violation_rate"]) > 0.05:
            lesson = "tail-risk token failed to prevent excessive penetration or force"
        elif float(row["slip_rate"]) > 0.35:
            lesson = "friction shift produced sliding chatter despite token adaptation"
        elif float(row["normalized_force_error"]) > 0.50:
            lesson = "force tracking failed under shifted stiffness and target force"
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
                "chatter_rate": row["chatter_rate"],
                "lesson": lesson,
            }
        )
    return rows


def plot_metric(summary: list[dict[str, str]], split_order: Sequence[str], methods: Sequence[str], metric: str, title: str, path: Path, ylim: tuple[float, float] | None = None) -> None:
    def value(row: dict[str, str], prefix: str) -> float:
        for key in (f"{prefix}_{metric}", f"{prefix}_mean_{metric}"):
            if key in row:
                return float(row[key])
        raise KeyError(metric)

    width = min(0.06, 0.70 / max(1, len(methods)))
    x = np.arange(len(split_order))
    plt.figure(figsize=(15, 5.5))
    for idx, method in enumerate(methods):
        vals = []
        errs = []
        for split in split_order:
            matches = [r for r in summary if r["method"] == method and r["split"] == split]
            if matches:
                row = matches[0]
                vals.append(value(row, "mean"))
                errs.append(value(row, "ci95"))
            else:
                vals.append(0.0)
                errs.append(0.0)
        plt.bar(x + (idx - len(methods) / 2) * width, vals, width, yerr=errs, label=method)
    plt.xticks(x, split_order, rotation=25, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.legend(fontsize=6, ncol=3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def plot_ablation(ablation_aggregate: list[dict[str, str]], path: Path) -> None:
    rows = sorted(ablation_aggregate, key=lambda row: float(row["success"]), reverse=True)
    plt.figure(figsize=(11, 5))
    plt.bar([row["method"] for row in rows], [float(row["success"]) for row in rows], color="#6d7f71")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("success rate")
    plt.title("Paper 72 impedance-token ablation aggregate")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def plot_stress(stress_summary: list[dict[str, str]], path: Path) -> None:
    plt.figure(figsize=(10, 5.5))
    rows = [row for row in stress_summary if row["split"] == "combined_extreme_stress"]
    for method in sorted({row["method"] for row in rows}):
        method_rows = sorted([row for row in rows if row["method"] == method], key=lambda r: float(r["stress_level"]))
        x = [float(row["stress_level"]) for row in method_rows]
        y = [float(row["mean_success_rate"]) for row in method_rows]
        e = [float(row["ci95_success_rate"]) for row in method_rows]
        plt.errorbar(x, y, yerr=e, marker="o", label=method)
    plt.xlabel("stress level")
    plt.ylabel("success rate")
    plt.title("Paper 72 combined-extreme stress sweep")
    plt.ylim(0, 1.0)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def decide(
    aggregate: list[dict[str, str]],
    group_pairwise: list[dict[str, str]],
    fixed_risk: list[dict[str, str]],
    ablation_aggregate: list[dict[str, str]],
    stress_summary: list[dict[str, str]],
) -> tuple[str, str]:
    proposed = "impedance_token_policy_v5"
    non_oracle = {proposed, "oracle_impedance"}
    hard_v5 = [row for row in aggregate if row["group"] == "hard_splits" and row["method"] == proposed][0]
    hard_best = max([row for row in aggregate if row["group"] == "hard_splits" and row["method"] not in non_oracle], key=lambda row: float(row["success_rate"]))
    combined_v5 = [row for row in aggregate if row["group"] == "combined_and_extreme" and row["method"] == proposed][0]
    combined_best = max([row for row in aggregate if row["group"] == "combined_and_extreme" and row["method"] not in non_oracle], key=lambda row: float(row["success_rate"]))
    hard_pair = [row for row in group_pairwise if row["group"] == "hard_splits" and row["comparison"] == hard_best["method"]][0]
    fixed_v5 = [row for row in fixed_risk if row["budget"] == "0.10" and row["method"] == proposed][0]
    fixed_best = max([row for row in fixed_risk if row["budget"] == "0.10" and row["method"] not in non_oracle], key=lambda row: float(row["success_at_budget"]))
    max_rows = [row for row in stress_summary if row["stress_level"] == "1.00"]
    max_v5 = float(np.mean([float(row["mean_success_rate"]) for row in max_rows if row["method"] == proposed]))
    max_best_row = max([row for row in max_rows if row["method"] not in non_oracle], key=lambda row: float(row["mean_success_rate"]))
    full_ablation = [row for row in ablation_aggregate if row["method"] == "token_full_v5"][0]
    bad_ablations = [
        row["method"]
        for row in ablation_aggregate
        if row["method"] != "token_full_v5" and float(row["success"]) >= float(full_ablation["success"]) - 1e-9
    ]
    failures = []
    if float(hard_v5["success_rate"]) < float(hard_best["success_rate"]) + 0.030:
        failures.append(
            f"v5 does not beat strongest hard-regime baseline {hard_best['method']} by 0.030 "
            f"(v5={float(hard_v5['success_rate']):.3f}, best={float(hard_best['success_rate']):.3f})"
        )
    if float(hard_pair["paired_success_diff"]) - float(hard_pair["ci95_success_diff"]) <= 0.0:
        failures.append(
            f"paired lower bound against {hard_best['method']} is not positive "
            f"({float(hard_pair['paired_success_diff']):.3f}+/-{float(hard_pair['ci95_success_diff']):.3f})"
        )
    if float(combined_v5["success_rate"]) < float(combined_best["success_rate"]) + 0.030:
        failures.append(
            f"v5 does not beat strongest combined/extreme baseline {combined_best['method']} by 0.030 "
            f"(v5={float(combined_v5['success_rate']):.3f}, best={float(combined_best['success_rate']):.3f})"
        )
    if float(fixed_v5["success_at_budget"]) < float(fixed_best["success_at_budget"]) + 0.030:
        failures.append(
            f"fixed-risk gate fails at budget 0.10 (v5={float(fixed_v5['success_at_budget']):.3f}, "
            f"best={fixed_best['method']} {float(fixed_best['success_at_budget']):.3f})"
        )
    if max_v5 < float(max_best_row["mean_success_rate"]) - 1e-9:
        failures.append(
            f"maximum-stress gate fails (v5={max_v5:.3f}, best={max_best_row['method']} {float(max_best_row['mean_success_rate']):.3f})"
        )
    if bad_ablations:
        failures.append("ablation gate fails because " + ", ".join(sorted(bad_ablations)) + " matches or beats full v5")
    if failures:
        return "KILL_ARCHIVE", "; ".join(failures)
    return (
        "STRONG_REVISE",
        "v5 clears hard-regime, paired, combined/extreme, fixed-risk, maximum-stress, and ablation-necessity gates, "
        "but still lacks real-robot/public-benchmark validation.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--ablation-episodes", type=int, default=DEFAULT_ABLATION_EPISODES)
    parser.add_argument("--stress-episodes", type=int, default=DEFAULT_STRESS_EPISODES)
    parser.add_argument("--train-scenes", type=int, default=DEFAULT_TRAINING_EXAMPLES)
    parser.add_argument("--splits", nargs="*", default=[split.name for split in SPLITS])
    parser.add_argument("--ablation-splits", nargs="*", default=["combined_stress", "combined_extreme_stress", "stick_slip_cycle", "actuator_saturation"])
    parser.add_argument("--stress-splits", nargs="*", default=["combined_stress", "combined_extreme_stress", "friction_slip_shift"])
    parser.add_argument("--stress-levels", nargs="*", type=float, default=[0.0, 0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--workers", type=int, default=1, help="Accepted for protocol compatibility; execution remains single-process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    results_dir: Path = args.results_dir
    figures_dir: Path = args.figures_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    split_by_name = {split.name: split for split in SPLITS}
    selected_splits = [split_by_name[name] for name in args.splits]
    ablation_splits = [split_by_name[name] for name in args.ablation_splits]
    stress_splits = [split_by_name[name] for name in args.stress_splits]

    pack = generate_training_pack(args.train_scenes)
    write_csv(results_dir / "training_impedance_examples.csv", pack.training_rows)
    write_csv(
        results_dir / "training_summary.csv",
        [
            {
                "training_examples": str(args.train_scenes),
                "ridge_train_mae": f"{pack.ridge_mae:.5f}",
                "random_forest_train_mae": f"{pack.forest_mae:.5f}",
                "hist_gradient_train_mae": f"{pack.hist_mae:.5f}",
                "conformal_margin": f"{pack.conformal_margin:.5f}",
                "feature_dim": str(len(pack.scaler_x.mean_)),
                "target_dim": "4",
            }
        ],
    )

    model = make_model()
    raw_rows: list[dict[str, str]] = []
    for split in selected_splits:
        for seed in seeds:
            for episode in range(args.episodes):
                cfg = make_config(split, seed, episode)
                for method in MAIN_METHODS:
                    raw_rows.append(simulate_episode(model, method, cfg, pack))
    write_csv(results_dir / "impedance_raw.csv", raw_rows)
    write_csv(results_dir / "impedance_rollouts.csv", raw_rows)
    seed_rows = build_seed_metrics(raw_rows)
    summary = build_summary(seed_rows)
    pairwise = build_pairwise(seed_rows)
    aggregate = build_aggregate(seed_rows)
    group_pairwise = build_group_pairwise(seed_rows)
    fixed_risk = build_fixed_risk(raw_rows)
    write_csv(results_dir / "raw_seed_metrics.csv", seed_rows)
    write_csv(results_dir / "metrics.csv", summary)
    write_csv(results_dir / "impedance_metrics.csv", summary)
    write_csv(results_dir / "pairwise_stats.csv", pairwise)
    write_csv(results_dir / "impedance_pairwise.csv", pairwise)
    write_csv(results_dir / "aggregate_metrics.csv", aggregate)
    write_csv(results_dir / "aggregate_pairwise_stats.csv", group_pairwise)
    write_csv(results_dir / "fixed_risk_metrics.csv", fixed_risk)

    ablation_raw: list[dict[str, str]] = []
    for split in ablation_splits:
        for seed in seeds:
            for episode in range(args.ablation_episodes):
                cfg = make_config(split, seed, 1000 + episode)
                for method in ABLATION_METHODS:
                    row = simulate_episode(model, method, cfg, pack)
                    row["method"] = method
                    ablation_raw.append(row)
    write_csv(results_dir / "impedance_ablation_raw.csv", ablation_raw)
    ablation_seed = build_seed_metrics(ablation_raw)
    ablation_summary = build_summary(ablation_seed)
    ablation_aggregate = build_ablation_aggregate(ablation_summary)
    write_csv(results_dir / "ablation_metrics.csv", ablation_summary)
    write_csv(results_dir / "impedance_ablation.csv", ablation_summary)
    write_csv(results_dir / "ablation_aggregate_metrics.csv", ablation_aggregate)

    stress_raw: list[dict[str, str]] = []
    for split in stress_splits:
        for stress_level in args.stress_levels:
            for seed in seeds:
                for episode in range(args.stress_episodes):
                    cfg = make_config(split, seed, 2000 + episode, stress_level=float(stress_level))
                    for method in STRESS_METHODS:
                        row = simulate_episode(model, method, cfg, pack)
                        row["split"] = split.name
                        row["stress_level"] = f"{float(stress_level):.2f}"
                        stress_raw.append(row)
    write_csv(results_dir / "stress_sweep_raw.csv", stress_raw)
    stress_summary = build_stress_summary(stress_raw, args.stress_episodes)
    write_csv(results_dir / "stress_sweep.csv", stress_summary)
    write_csv(figures_dir / "stress_curve_data.csv", stress_summary)
    write_csv(results_dir / "negative_cases.csv", negative_cases(raw_rows))

    split_order = [split.name for split in selected_splits]
    plot_methods = [
        "gain_scheduled_impedance",
        "learned_gain_regressor",
        "random_forest_gain_regressor",
        "hist_gradient_gain_regressor",
        "ensemble_uncertainty_gain",
        "risk_averse_impedance",
        "conformal_safety_gain",
        "impedance_token_policy_v4",
        "impedance_token_policy_v5",
        "oracle_impedance",
    ]
    plot_metric(summary, split_order, plot_methods, "success_rate", "Paper 72 MuJoCo impedance-token success", figures_dir / "impedance_success_by_split.png", (0, 1.0))
    plot_metric(summary, split_order, plot_methods, "normalized_force_error", "Paper 72 normalized force tracking error", figures_dir / "impedance_force_error_by_split.png")
    plot_metric(summary, split_order, plot_methods, "safety_violation_rate", "Paper 72 safety violation rate", figures_dir / "impedance_safety_by_split.png", (0, 0.45))
    plot_ablation(ablation_aggregate, figures_dir / "impedance_ablation_success.png")
    plot_stress(stress_summary, figures_dir / "impedance_stress_sweep.png")

    decision, reason = decide(aggregate, group_pairwise, fixed_risk, ablation_aggregate, stress_summary)
    elapsed = time.time() - start_time
    combined_rows = [r for r in aggregate if r["group"] == "combined_and_extreme"]
    with (results_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Paper 72 adaptive_impedance_tokens expanded MuJoCo rebuild\n")
        f.write(f"Terminal decision: {decision}\n")
        f.write(f"Terminal reason: {reason}\n")
        f.write(f"Main eval rows: {len(raw_rows)}\n")
        f.write(f"Ablation rows: {len(ablation_raw)}\n")
        f.write(f"Stress rows: {len(stress_raw)}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Eval episodes per seed/split: {args.episodes}\n")
        f.write(f"Runtime seconds: {elapsed:.2f}\n\n")
        f.write("Combined/extreme aggregate summary:\n")
        for row in sorted(combined_rows, key=lambda r: -float(r["success_rate"])):
            f.write(
                f"{row['method']} success={row['success_rate']} ci95={row['ci95_success_rate']} "
                f"force_error={row['normalized_force_error']} safety={row['safety_violation_rate']} "
                f"slip={row['slip_rate']} chatter={row['chatter_rate']}\n"
            )
    print(f"wrote Paper 72 expanded MuJoCo impedance evidence to {results_dir}")
    print(f"terminal decision: {decision}")
    print(f"terminal reason: {reason}")


if __name__ == "__main__":
    main()
