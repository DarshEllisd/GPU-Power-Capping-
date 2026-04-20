from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from archived.rl_api import SafetyShield


def load_analysis(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"clock_applied_mhz", "slowdown_pct", "energy_saving_pct"}
    missing = need.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def build_obs(workloads: list[str], workload: str, budget: float, current_clock: int, clocks: list[int]) -> np.ndarray:
    onehot = np.zeros(len(workloads), dtype=np.float32)
    onehot[workloads.index(workload)] = 1.0
    min_clock = float(min(clocks))
    max_clock = float(max(clocks))
    norm_clock = (float(current_clock) - min_clock) / max(max_clock - min_clock, 1.0)
    norm_budget = float(budget) / 30.0
    return np.concatenate([onehot, np.asarray([norm_budget, norm_clock], dtype=np.float32)])


def build_v4_obs(
    budget: float,
    current_clock: int,
    clocks: list[int],
    util_mean: float,
    util_std: float,
    power_w: float,
    burst_sec: float,
) -> np.ndarray:
    min_clock = float(min(clocks))
    max_clock = float(max(clocks))
    norm_clock = (float(current_clock) - min_clock) / max(max_clock - min_clock, 1.0)
    norm_budget = float(budget) / 30.0
    return np.asarray(
        [
            float(np.clip(util_mean / 100.0, 0.0, 1.0)),
            float(np.clip(util_std / 100.0, 0.0, 1.0)),
            float(np.clip(power_w / 120.0, 0.0, 1.0)),
            float(np.clip(burst_sec / 8.0, 0.0, 1.0)),
            float(np.clip(norm_budget, 0.0, 1.0)),
            float(np.clip(norm_clock, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_cap_policy", help="path to PPO model")
    parser.add_argument("--meta", default="models/ppo_cap_policy_meta.json", help="path to metadata json")
    parser.add_argument("--analysis", required=True, help="analysis csv for selected workload")
    parser.add_argument("--workload", choices=["compute", "memory", "mixed"], help="workload label for v3-style models")
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--current-clock", type=int, default=2100)
    parser.add_argument("--max-jump", type=int, default=120)
    parser.add_argument("--guard-band", type=float, default=0.5)
    parser.add_argument("--util-mean", type=float, default=50.0, help="v4 telemetry input")
    parser.add_argument("--util-std", type=float, default=20.0, help="v4 telemetry input")
    parser.add_argument("--power-w", type=float, default=25.0, help="v4 telemetry input")
    parser.add_argument("--burst-sec", type=float, default=1.0, help="v4 telemetry input")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = PPO.load(args.model, device="cpu")
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))

    workloads = [str(w) for w in meta.get("workloads", [])]
    clock_values = [int(c) for c in meta["clock_values"]]

    model_type = str(meta.get("model_type", "v3_workload_onehot"))
    use_v4 = model_type.startswith("v4") or meta.get("obs_fields") == [
        "util_mean_norm",
        "util_std_norm",
        "power_norm",
        "burst_norm",
        "budget_norm",
        "current_clock_norm",
    ]

    if use_v4:
        obs = build_v4_obs(
            budget=args.budget,
            current_clock=args.current_clock,
            clocks=clock_values,
            util_mean=args.util_mean,
            util_std=args.util_std,
            power_w=args.power_w,
            burst_sec=args.burst_sec,
        )
    else:
        if not args.workload:
            raise ValueError("--workload is required for non-v4 models")
        if args.workload not in workloads:
            raise ValueError(f"workload '{args.workload}' not in model metadata workloads={workloads}")

        obs = build_obs(
            workloads=workloads,
            workload=args.workload,
            budget=args.budget,
            current_clock=args.current_clock,
            clocks=clock_values,
        )

    action, _ = model.predict(obs, deterministic=True)
    proposed_clock = int(clock_values[int(action)])

    analysis = load_analysis(Path(args.analysis))
    slowdown_by_clock = {
        int(r["clock_applied_mhz"]): float(r["slowdown_pct"])
        for _, r in analysis.iterrows()
    }

    fallback = int(
        analysis.sort_values(["slowdown_pct", "energy_saving_pct"], ascending=[True, False]).iloc[0][
            "clock_applied_mhz"
        ]
    )

    shield = SafetyShield(
        supported_clocks_mhz=clock_values,
        max_jump_mhz=args.max_jump,
        slowdown_budget_pct=args.budget,
        guard_band_pct=args.guard_band,
    )
    safe_clock = shield.apply(
        proposed_clock_mhz=proposed_clock,
        current_clock_mhz=args.current_clock,
        predicted_slowdown_by_clock=slowdown_by_clock,
        fallback_clock_mhz=fallback,
    )

    row = analysis[analysis["clock_applied_mhz"] == safe_clock].iloc[0]
    print(f"model_type={model_type}")
    if not use_v4:
        print(f"workload={args.workload}")
    print(f"budget_pct={args.budget:.2f}")
    print(f"current_clock={args.current_clock}")
    print(f"proposed_clock={proposed_clock}")
    print(f"safe_clock={safe_clock}")
    print(f"predicted_slowdown_pct={float(row['slowdown_pct']):.3f}")
    print(f"predicted_energy_saving_pct={float(row['energy_saving_pct']):.3f}")


if __name__ == "__main__":
    main()
