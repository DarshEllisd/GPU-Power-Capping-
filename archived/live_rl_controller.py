from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from archived.rl_api import SafetyShield, select_clock_under_budget


@dataclass
class Sample:
    gpu_util: float
    power_w: float
    clock_mhz: int
    temp_c: float
    ts: float


class QoSProxy:
    """
    Execution-time proxy built from active GPU burst durations.
    No rollback logic is used here; this only estimates slowdown proxy for observability.
    """

    def __init__(self, util_active_threshold: float = 70.0, alpha: float = 0.2) -> None:
        self.util_active_threshold = util_active_threshold
        self.alpha = alpha
        self.active = False
        self.burst_start = 0.0
        self.burst_ewma_sec = 0.0

    def update(self, util: float, now: float) -> float:
        if util >= self.util_active_threshold:
            if not self.active:
                self.active = True
                self.burst_start = now
            burst = now - self.burst_start
            if self.burst_ewma_sec <= 1e-6:
                self.burst_ewma_sec = max(burst, 1e-3)
            else:
                self.burst_ewma_sec = (1.0 - self.alpha) * self.burst_ewma_sec + self.alpha * max(burst, 1e-3)
        else:
            if self.active:
                burst = now - self.burst_start
                if self.burst_ewma_sec <= 1e-6:
                    self.burst_ewma_sec = max(burst, 1e-3)
                else:
                    self.burst_ewma_sec = (1.0 - self.alpha) * self.burst_ewma_sec + self.alpha * max(burst, 1e-3)
            self.active = False
        return max(self.burst_ewma_sec, 1e-3)


def run_checked(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def query_sample(gpu_index: int) -> Sample:
    result = run_checked(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,power.draw,clocks.gr,temperature.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_index),
        ]
    )
    line = result.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    util = float(parts[0])
    power = float(parts[1])
    clock = int(float(parts[2]))
    temp = float(parts[3])
    return Sample(gpu_util=util, power_w=power, clock_mhz=clock, temp_c=temp, ts=time.time())


def apply_cap(gpu_index: int, clock_mhz: int) -> None:
    run_checked(["nvidia-smi", "-i", str(gpu_index), "-lgc", f"{clock_mhz},{clock_mhz}"])


def reset_cap(gpu_index: int) -> None:
    run_checked(["nvidia-smi", "-i", str(gpu_index), "-rgc"])


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def classify_workload_probs(util_mean: float, util_std: float, burst_ewma_sec: float) -> dict[str, float]:
    """
    Soft workload classifier that returns probabilities instead of a hard label.
    This avoids brittle cap swings caused by hard workload boundaries.
    """
    score_compute = _sigmoid((util_mean - 88.0) / 7.0) + _sigmoid((burst_ewma_sec - 1.5) / 0.35)
    score_memory = _sigmoid((70.0 - util_mean) / 7.0) + _sigmoid((burst_ewma_sec - 1.0) / 0.35)
    score_mixed = _sigmoid((util_std - 18.0) / 6.0) + 0.20

    scores = {
        "compute": max(score_compute, 1e-6),
        "memory": max(score_memory, 1e-6),
        "mixed": max(score_mixed, 1e-6),
    }
    total = sum(scores.values())
    return {k: v / total for k, v in scores.items()}


def build_obs(workloads: list[str], workload_probs: dict[str, float], budget: float, current_clock: int, clocks: list[int]) -> np.ndarray:
    onehot = np.zeros(len(workloads), dtype=np.float32)
    for i, w in enumerate(workloads):
        onehot[i] = float(workload_probs.get(w, 0.0))
    min_clock = float(min(clocks))
    max_clock = float(max(clocks))
    norm_clock = (float(current_clock) - min_clock) / max(max_clock - min_clock, 1.0)
    norm_budget = float(budget) / 30.0
    return np.concatenate([onehot, np.asarray([norm_budget, norm_clock], dtype=np.float32)])


def build_v4_obs(budget: float, current_clock: int, clocks: list[int], util_mean: float, util_std: float, power_mean: float, burst_ewma: float) -> np.ndarray:
    min_clock = float(min(clocks))
    max_clock = float(max(clocks))
    norm_clock = (float(current_clock) - min_clock) / max(max_clock - min_clock, 1.0)
    norm_budget = float(budget) / 30.0
    return np.asarray(
        [
            float(np.clip(util_mean / 100.0, 0.0, 1.0)),
            float(np.clip(util_std / 100.0, 0.0, 1.0)),
            float(np.clip(power_mean / 120.0, 0.0, 1.0)),
            float(np.clip(burst_ewma / 8.0, 0.0, 1.0)),
            float(np.clip(norm_budget, 0.0, 1.0)),
            float(np.clip(norm_clock, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def nearest_clock(target: int, clocks: list[int]) -> int:
    return int(min(clocks, key=lambda c: abs(c - target)))


def clamp_jump(current: int, target: int, max_jump: int) -> int:
    if target > current + max_jump:
        return current + max_jump
    if target < current - max_jump:
        return current - max_jump
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_cap_policy")
    parser.add_argument("--meta", default="models/ppo_cap_policy_meta.json")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--max-jump", type=int, default=120)
    parser.add_argument("--guard-band-pct", type=float, default=0.5)
    parser.add_argument("--util-active-threshold", type=float, default=70.0)
    parser.add_argument("--baseline-latency-sec", type=float, default=0.0)
    parser.add_argument("--analysis-compute", default="data/processed/compute_vs_baseline_analysis.csv")
    parser.add_argument("--analysis-memory", default="data/processed/memory_vs_baseline_analysis.csv")
    parser.add_argument("--analysis-mixed", default="data/processed/mixed_vs_baseline_analysis.csv")
    parser.add_argument("--dry-run", action="store_true", help="Compute decisions and log them without applying clocks")
    parser.add_argument("--reset-on-exit", action="store_true")
    return parser.parse_args()


def load_analysis_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(Path(path))
    required = {"clock_applied_mhz", "slowdown_pct", "energy_saving_pct"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df.sort_values("clock_applied_mhz").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    model = PPO.load(args.model, device="cpu")
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))

    workloads = [str(w) for w in meta["workloads"]]
    clocks = [int(c) for c in meta["clock_values"]]

    analysis_paths = {
        "compute": args.analysis_compute,
        "memory": args.analysis_memory,
        "mixed": args.analysis_mixed,
    }
    analysis_tables: dict[str, pd.DataFrame] = {}
    slowdown_by_workload: dict[str, dict[int, float]] = {}
    fallback_by_workload: dict[str, int] = {}

    try:
        for workload in workloads:
            if workload not in analysis_paths:
                continue
            table = load_analysis_table(analysis_paths[workload])
            analysis_tables[workload] = table
            slowdown_by_workload[workload] = {
                int(r["clock_applied_mhz"]): float(r["slowdown_pct"])
                for _, r in table.iterrows()
            }
            fallback_by_workload[workload] = int(
                select_clock_under_budget(table, slowdown_budget_pct=args.budget)
            )
    except Exception as exc:
        print(f"Warning: failed to load analysis tables for budget shielding: {exc}", flush=True)
        analysis_tables = {}
        slowdown_by_workload = {}
        fallback_by_workload = {}

    shield = SafetyShield(
        supported_clocks_mhz=clocks,
        max_jump_mhz=args.max_jump,
        slowdown_budget_pct=args.budget,
        guard_band_pct=args.guard_band_pct,
    )

    model_type = str(meta.get("model_type", "v3_workload_onehot"))
    use_v4 = model_type.startswith("v4") or meta.get("obs_fields") == [
        "util_mean_norm",
        "util_std_norm",
        "power_norm",
        "burst_norm",
        "budget_norm",
        "current_clock_norm",
    ]

    qos = QoSProxy(util_active_threshold=args.util_active_threshold, alpha=0.2)
    window: deque[Sample] = deque(maxlen=max(args.window_size, 3))
    current_cap: int | None = None

    print("Starting live RL controller (no rollback safety mode)...", flush=True)
    print(f"budget={args.budget:.2f}% | interval={args.interval_sec:.2f}s | max_jump={args.max_jump} MHz", flush=True)

    try:
        while True:
            s = query_sample(args.gpu_index)
            window.append(s)

            burst_ewma = qos.update(s.gpu_util, s.ts)
            util_values = np.asarray([x.gpu_util for x in window], dtype=np.float32)
            power_values = np.asarray([x.power_w for x in window], dtype=np.float32)

            util_mean = float(util_values.mean()) if len(util_values) else 0.0
            util_std = float(util_values.std()) if len(util_values) else 0.0
            power_mean = float(power_values.mean()) if len(power_values) else 0.0

            workload_probs = classify_workload_probs(
                util_mean=util_mean,
                util_std=util_std,
                burst_ewma_sec=burst_ewma,
            )
            dominant_workload = max(workload_probs.items(), key=lambda kv: kv[1])[0]
            if use_v4:
                obs = build_v4_obs(
                    budget=args.budget,
                    current_clock=s.clock_mhz,
                    clocks=clocks,
                    util_mean=util_mean,
                    util_std=util_std,
                    power_mean=power_mean,
                    burst_ewma=burst_ewma,
                )
            else:
                obs = build_obs(workloads, workload_probs, args.budget, s.clock_mhz, clocks)

            if current_cap is None:
                current_cap = int(s.clock_mhz)

            action, _ = model.predict(obs, deterministic=True)
            proposed = int(clocks[int(action)])

            # Keep transitions smooth, then enforce generic budget-aware safety filtering if analysis is available.
            proposed = clamp_jump(current=current_cap, target=proposed, max_jump=args.max_jump)
            proposed = nearest_clock(proposed, clocks)

            slowdown_map: dict[int, float] | None = None
            fallback_clock: int | None = None
            if slowdown_by_workload:
                blended: dict[int, float] = {}
                for c in clocks:
                    val = 0.0
                    have_all = True
                    for w in workloads:
                        m = slowdown_by_workload.get(w)
                        if m is None or c not in m:
                            have_all = False
                            break
                        val += workload_probs.get(w, 0.0) * float(m[c])
                    if have_all:
                        blended[c] = val
                if blended:
                    slowdown_map = blended
                    safe_limit = args.budget - args.guard_band_pct
                    safe_candidates = [c for c, sdp in blended.items() if sdp <= safe_limit]
                    if safe_candidates:
                        fallback_clock = max(safe_candidates)
                    else:
                        fallback_clock = min(blended, key=lambda c: blended[c])

            if slowdown_map is not None and fallback_clock is not None:
                proposed = shield.apply(
                    proposed_clock_mhz=proposed,
                    current_clock_mhz=current_cap,
                    predicted_slowdown_by_clock=slowdown_map,
                    fallback_clock_mhz=fallback_clock,
                )
                proposed = nearest_clock(proposed, clocks)

            if not args.dry_run:
                apply_cap(args.gpu_index, proposed)
            current_cap = proposed

            if args.baseline_latency_sec > 1e-6:
                slowdown_proxy = ((burst_ewma / args.baseline_latency_sec) - 1.0) * 100.0
            else:
                slowdown_proxy = float("nan")

            print(
                " | ".join(
                    [
                        f"util={util_mean:.1f}%",
                        f"util_std={util_std:.1f}",
                        f"power={power_mean:.1f}W",
                        f"burst_ewma={burst_ewma:.3f}s",
                        f"slowdown_proxy={slowdown_proxy:.2f}%",
                        f"model_type={model_type}",
                        f"workload={dominant_workload}",
                        f"probs=c:{workload_probs.get('compute', 0.0):.2f},m:{workload_probs.get('memory', 0.0):.2f},x:{workload_probs.get('mixed', 0.0):.2f}",
                        f"cap={proposed}MHz",
                        f"budget={args.budget:.2f}%",
                        f"budget_shield={bool(slowdown_map is not None and fallback_clock is not None)}",
                        f"dry_run={args.dry_run}",
                    ]
                )
            , flush=True)

            time.sleep(args.interval_sec)

    except KeyboardInterrupt:
        print("Stopping live controller...", flush=True)
    finally:
        if args.reset_on_exit:
            try:
                reset_cap(args.gpu_index)
                print("Clocks reset to default.", flush=True)
            except Exception as exc:
                print(f"Warning: failed to reset clocks: {exc}", flush=True)


if __name__ == "__main__":
    main()
