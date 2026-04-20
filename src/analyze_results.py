from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate_by_clock(df: pd.DataFrame) -> pd.DataFrame:
    agg_spec = {
        "runtime_s": ("runtime_s", "mean"),
        "avg_power_w": ("avg_power_w", "mean"),
        "energy_j": ("energy_j", "mean"),
        "repeats": ("run_id", "count"),
    }
    if "avg_clock_mhz" in df.columns:
        agg_spec["avg_clock_mhz"] = ("avg_clock_mhz", "mean")
    if "peak_clock_mhz" in df.columns:
        agg_spec["peak_clock_mhz"] = ("peak_clock_mhz", "mean")
    if "avg_temp_c" in df.columns:
        agg_spec["avg_temp_c"] = ("avg_temp_c", "mean")
    if "avg_gpu_util_pct" in df.columns:
        agg_spec["avg_gpu_util_pct"] = ("avg_gpu_util_pct", "mean")
    if "avg_mem_bw_util_pct" in df.columns:
        agg_spec["avg_mem_bw_util_pct"] = ("avg_mem_bw_util_pct", "mean")
    if "avg_vram_used_mib" in df.columns:
        agg_spec["avg_vram_used_mib"] = ("avg_vram_used_mib", "mean")
    if "avg_vram_alloc_pct" in df.columns:
        agg_spec["avg_vram_alloc_pct"] = ("avg_vram_alloc_pct", "mean")

    grouped = df.groupby("clock_applied_mhz", as_index=False).agg(**agg_spec).sort_values(
        "clock_applied_mhz", ascending=False
    )
    return grouped


def add_relative_metrics(df: pd.DataFrame, baseline_runtime: float | None = None, baseline_energy: float | None = None) -> pd.DataFrame:
    if baseline_runtime is None or baseline_energy is None:
        baseline = df.iloc[0]
        t0 = float(baseline["runtime_s"])
        e0 = float(baseline["energy_j"])
    else:
        t0 = baseline_runtime
        e0 = baseline_energy

    out = df.copy()
    out["slowdown_pct"] = (out["runtime_s"] / t0 - 1.0) * 100.0
    out["energy_saving_pct"] = (1.0 - out["energy_j"] / e0) * 100.0
    out["edp"] = out["energy_j"] * out["runtime_s"]
    out["norm_edp"] = out["edp"] / float((e0 * t0))
    return out


def pick_recommended_clock(df: pd.DataFrame, max_slowdown_pct: float) -> int:
    feasible = df[df["slowdown_pct"] <= max_slowdown_pct]
    if feasible.empty:
        return int(df.iloc[0]["clock_applied_mhz"])

    best = feasible.sort_values(
        ["energy_saving_pct", "runtime_s"], ascending=[False, True]
    ).iloc[0]
    return int(best["clock_applied_mhz"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to summary.csv")
    parser.add_argument("--output", required=True, help="Path to analysis CSV")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional: Path to baseline summary.csv (to use as reference instead of first row)",
    )
    parser.add_argument(
        "--max-slowdown-pct",
        type=float,
        default=5.0,
        help="Runtime loss cap for recommendation",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(inp)
    required = {
        "run_id",
        "clock_applied_mhz",
        "avg_clock_mhz",
        "peak_clock_mhz",
        "avg_temp_c",
        "avg_gpu_util_pct",
        "avg_mem_bw_util_pct",
        "avg_vram_used_mib",
        "avg_vram_alloc_pct",
        "runtime_s",
        "avg_power_w",
        "energy_j",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Load baseline if provided
    baseline_runtime = None
    baseline_energy = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline file not found: {baseline_path}")
        baseline_raw = pd.read_csv(baseline_path)
        baseline_agg = aggregate_by_clock(baseline_raw)
        baseline_row = baseline_agg.iloc[0]
        baseline_runtime = float(baseline_row["runtime_s"])
        baseline_energy = float(baseline_row["energy_j"])

    agg = aggregate_by_clock(raw)
    analysis = add_relative_metrics(agg, baseline_runtime, baseline_energy)
    rec = pick_recommended_clock(analysis, args.max_slowdown_pct)

    analysis.to_csv(out, index=False)

    min_edp_row = analysis.loc[analysis["edp"].idxmin()]
    min_power_row = analysis.loc[analysis["avg_power_w"].idxmin()]

    print(f"Saved analysis: {out}")
    print(f"Recommended clock under {args.max_slowdown_pct:.1f}% slowdown cap: {rec} MHz")
    print(
        "Min EDP clock: "
        f"{int(min_edp_row['clock_applied_mhz'])} MHz "
        f"(norm_edp={float(min_edp_row['norm_edp']):.4f})"
    )
    print(
        "Min avg power clock: "
        f"{int(min_power_row['clock_applied_mhz'])} MHz "
        f"(avg_power={float(min_power_row['avg_power_w']):.2f} W)"
    )


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    pd.set_option("display.width", 140)
    main()
