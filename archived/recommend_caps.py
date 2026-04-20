from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from archived.rl_api import select_clock_under_budget


def parse_budgets(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        values.append(float(p))
    if not values:
        raise ValueError("At least one budget must be provided")
    return values


def summarize_best(df: pd.DataFrame, clock: int) -> tuple[float, float, float]:
    row = df[df["clock_applied_mhz"] == clock].iloc[0]
    return (
        float(row["slowdown_pct"]),
        float(row["energy_saving_pct"]),
        float(row["norm_edp"]),
    )


def run_for_file(path: Path, budgets: list[float]) -> None:
    df = pd.read_csv(path)
    print(f"\n=== {path.stem} ===")
    for budget in budgets:
        clock = select_clock_under_budget(df, budget)
        slowdown, energy_saving, norm_edp = summarize_best(df, clock)
        print(
            f"budget={budget:.1f}% -> clock={clock} MHz | "
            f"slowdown={slowdown:.2f}% | energy_saving={energy_saving:.2f}% | norm_edp={norm_edp:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more analysis CSV files",
    )
    parser.add_argument(
        "--budgets",
        default="2,5,10",
        help="Comma-separated slowdown budgets in percent",
    )
    args = parser.parse_args()

    budgets = parse_budgets(args.budgets)
    for inp in args.inputs:
        run_for_file(Path(inp), budgets)


if __name__ == "__main__":
    main()
