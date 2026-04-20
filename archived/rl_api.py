from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TelemetryWindow:
    gpu_util_mean: float
    gpu_util_std: float
    mem_util_mean: float
    mem_util_std: float
    power_mean_w: float
    power_trend_w_per_s: float
    temp_c: float
    current_clock_mhz: int


@dataclass(frozen=True)
class RewardConfig:
    slowdown_budget_pct: float = 5.0
    slowdown_penalty: float = 1.0
    jitter_penalty: float = 0.05
    energy_scale: float = 50.0
    violation_scale: float = 10.0
    jump_scale_mhz: float = 120.0


def build_state_vector(
    telemetry: TelemetryWindow,
    min_clock_mhz: int,
    max_clock_mhz: int,
    slowdown_budget_pct: float,
    workload_onehot: Iterable[float],
) -> np.ndarray:
    """Build an app-agnostic RL state vector for online frequency capping."""
    clock_denom = max(float(max_clock_mhz - min_clock_mhz), 1.0)
    norm_clock = (telemetry.current_clock_mhz - min_clock_mhz) / clock_denom

    state = [
        np.clip(telemetry.gpu_util_mean / 100.0, 0.0, 1.0),
        np.clip(telemetry.gpu_util_std / 100.0, 0.0, 1.0),
        np.clip(telemetry.mem_util_mean / 100.0, 0.0, 1.0),
        np.clip(telemetry.mem_util_std / 100.0, 0.0, 1.0),
        np.clip(telemetry.power_mean_w / 400.0, 0.0, 1.0),
        np.clip((telemetry.power_trend_w_per_s + 50.0) / 100.0, 0.0, 1.0),
        np.clip(telemetry.temp_c / 100.0, 0.0, 1.0),
        np.clip(norm_clock, 0.0, 1.0),
        np.clip(slowdown_budget_pct / 30.0, 0.0, 1.0),
    ]

    state.extend(float(v) for v in workload_onehot)
    return np.asarray(state, dtype=np.float32)


def compute_reward(
    energy_saving_pct: float,
    slowdown_pct: float,
    clock_jump_mhz: int,
    cfg: RewardConfig,
) -> float:
    """
    Reward function for constrained optimization:
      maximize energy saving while respecting slowdown budget.
    """
    violation = max(0.0, slowdown_pct - cfg.slowdown_budget_pct)
    scaled_energy = float(energy_saving_pct) / max(cfg.energy_scale, 1e-6)
    scaled_violation = violation / max(cfg.violation_scale, 1e-6)
    scaled_jump = abs(float(clock_jump_mhz)) / max(cfg.jump_scale_mhz, 1e-6)
    return (
        scaled_energy
        - cfg.slowdown_penalty * (scaled_violation**2)
        - cfg.jitter_penalty * scaled_jump
    )


class SafetyShield:
    """Safety filter for action proposals from RL policy."""

    def __init__(
        self,
        supported_clocks_mhz: Iterable[int],
        max_jump_mhz: int = 120,
        slowdown_budget_pct: float = 5.0,
        guard_band_pct: float = 0.5,
    ) -> None:
        self.supported_clocks = sorted({int(c) for c in supported_clocks_mhz}, reverse=True)
        self.max_jump_mhz = int(max_jump_mhz)
        self.slowdown_budget_pct = float(slowdown_budget_pct)
        self.guard_band_pct = float(guard_band_pct)

    def allowed_actions(self, current_clock_mhz: int) -> list[int]:
        lo = current_clock_mhz - self.max_jump_mhz
        hi = current_clock_mhz + self.max_jump_mhz
        return [c for c in self.supported_clocks if lo <= c <= hi]

    def apply(
        self,
        proposed_clock_mhz: int,
        current_clock_mhz: int,
        predicted_slowdown_by_clock: dict[int, float],
        fallback_clock_mhz: int,
    ) -> int:
        candidates = self.allowed_actions(current_clock_mhz)
        safe_limit = self.slowdown_budget_pct - self.guard_band_pct

        safe = [
            c
            for c in candidates
            if predicted_slowdown_by_clock.get(c, np.inf) <= safe_limit
        ]
        if not safe:
            return int(fallback_clock_mhz)

        if proposed_clock_mhz in safe:
            return int(proposed_clock_mhz)

        return int(min(safe, key=lambda c: abs(c - proposed_clock_mhz)))


def select_clock_under_budget(
    analysis_df: pd.DataFrame,
    slowdown_budget_pct: float,
) -> int:
    """
    Pick the clock with max energy_saving_pct among rows satisfying slowdown budget.
    Falls back to lowest slowdown if no feasible row exists.
    """
    required = {"clock_applied_mhz", "slowdown_pct", "energy_saving_pct"}
    missing = required.difference(analysis_df.columns)
    if missing:
        raise ValueError(f"analysis_df missing columns: {sorted(missing)}")

    feasible = analysis_df[analysis_df["slowdown_pct"] <= slowdown_budget_pct]
    if feasible.empty:
        best = analysis_df.sort_values(["slowdown_pct", "energy_saving_pct"], ascending=[True, False]).iloc[0]
        return int(best["clock_applied_mhz"])

    best = feasible.sort_values(["energy_saving_pct", "slowdown_pct"], ascending=[False, True]).iloc[0]
    return int(best["clock_applied_mhz"])
