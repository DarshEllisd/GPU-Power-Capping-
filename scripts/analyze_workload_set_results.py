from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analyze_results import add_relative_metrics, aggregate_by_clock, pick_recommended_clock


METRIC_COLUMNS = [
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
    "repeats",
    "slowdown_pct",
    "energy_saving_pct",
    "edp",
    "norm_edp",
]


RAW_REQUIRED_COLUMNS = {
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


def analyze_one(
    capped_csv: Path,
    baseline_csv: Path,
    out_csv: Path,
    max_slowdown_pct: float,
) -> tuple[int, float, float, float, pd.DataFrame]:
    raw = pd.read_csv(capped_csv)
    baseline_raw = pd.read_csv(baseline_csv)

    raw_missing = sorted(RAW_REQUIRED_COLUMNS.difference(raw.columns))
    if raw_missing:
        raise ValueError(f"{capped_csv} missing required new-format columns: {raw_missing}")

    baseline_missing = sorted(RAW_REQUIRED_COLUMNS.difference(baseline_raw.columns))
    if baseline_missing:
        raise ValueError(f"{baseline_csv} missing required new-format columns: {baseline_missing}")

    agg = aggregate_by_clock(raw)
    baseline_agg = aggregate_by_clock(baseline_raw)

    baseline_row = baseline_agg.iloc[0]
    baseline_runtime = float(baseline_row["runtime_s"])
    baseline_energy = float(baseline_row["energy_j"])

    analysis = add_relative_metrics(agg, baseline_runtime, baseline_energy)
    recommended = pick_recommended_clock(analysis, max_slowdown_pct)

    ordered_cols = [c for c in METRIC_COLUMNS if c in analysis.columns]
    extra_cols = [c for c in analysis.columns if c not in ordered_cols]
    analysis = analysis[ordered_cols + extra_cols]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(out_csv, index=False)

    rec_row = analysis.loc[analysis["clock_applied_mhz"] == recommended].iloc[0]
    rec_slowdown = float(rec_row["slowdown_pct"])
    rec_energy_saving = float(rec_row["energy_saving_pct"])

    return recommended, rec_slowdown, rec_energy_saving, baseline_runtime, analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--categories", default="compute,memory")
    parser.add_argument("--max-slowdown-pct", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)
    categories = [x.strip() for x in args.categories.split(",") if x.strip()]

    summary_rows: list[list[object]] = []
    all_points_rows: list[pd.DataFrame] = []
    missing_pairs = 0
    processed_count = 0

    for category in categories:
        capped_dir = raw_root / category
        baseline_dir = capped_dir / "baseline"
        out_dir = processed_root / category

        if not capped_dir.exists():
            print(f"Skipping missing category dir: {capped_dir}")
            continue
        if not baseline_dir.exists():
            print(f"Skipping category with missing baseline dir: {baseline_dir}")
            continue

        workload_csvs = sorted(p for p in capped_dir.glob("*.csv") if p.is_file())
        for capped_csv in workload_csvs:
            baseline_csv = baseline_dir / capped_csv.name
            if not baseline_csv.exists():
                print(f"Missing baseline for {capped_csv.name}: expected {baseline_csv}")
                missing_pairs += 1
                continue

            out_csv = out_dir / f"{capped_csv.stem}_analysis.csv"
            recommended, rec_slowdown, rec_energy_saving, baseline_runtime, analysis = analyze_one(
                capped_csv=capped_csv,
                baseline_csv=baseline_csv,
                out_csv=out_csv,
                max_slowdown_pct=args.max_slowdown_pct,
            )

            tagged = analysis.copy()
            tagged.insert(0, "workload", capped_csv.stem)
            tagged.insert(0, "category", category)
            all_points_rows.append(tagged)

            processed_count += 1
            summary_rows.append(
                [
                    category,
                    capped_csv.stem,
                    str(capped_csv),
                    str(baseline_csv),
                    str(out_csv),
                    recommended,
                    f"{rec_slowdown:.4f}",
                    f"{rec_energy_saving:.4f}",
                    f"{baseline_runtime:.4f}",
                ]
            )
            print(
                f"[{category}/{capped_csv.stem}] -> {out_csv} | rec={recommended} MHz "
                f"slowdown={rec_slowdown:.2f}% energy_saving={rec_energy_saving:.2f}%"
            )

    summary_csv = processed_root / "workload_set_analysis_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "category",
            "workload",
            "capped_csv",
            "baseline_csv",
            "analysis_csv",
            "recommended_clock_mhz",
            "recommended_slowdown_pct",
            "recommended_energy_saving_pct",
            "baseline_runtime_s",
        ],
    )
    summary_df.to_csv(summary_csv, index=False)

    all_points_csv = processed_root / "workload_set_all_points.csv"
    all_points_csv.parent.mkdir(parents=True, exist_ok=True)
    if all_points_rows:
        all_points_df = pd.concat(all_points_rows, ignore_index=True)
        ordered_cols = ["category", "workload"] + METRIC_COLUMNS
        existing_ordered = [c for c in ordered_cols if c in all_points_df.columns]
        extra_cols = [c for c in all_points_df.columns if c not in existing_ordered]
        all_points_df = all_points_df[existing_ordered + extra_cols]
        all_points_df.to_csv(all_points_csv, index=False)
    else:
        pd.DataFrame(columns=["category", "workload"] + METRIC_COLUMNS).to_csv(all_points_csv, index=False)

    print(f"\nProcessed workload analyses: {processed_count}")
    print(f"Missing baseline pairs: {missing_pairs}")
    print(f"Summary: {summary_csv}")
    print(f"All points: {all_points_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
