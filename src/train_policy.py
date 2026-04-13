from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="CSV with workload features")
    parser.add_argument("--labels", required=True, help="CSV with optimal clock label")
    parser.add_argument("--output", required=True, help="Path to save policy model")
    parser.add_argument("--target-col", default="recommended_clock_mhz")
    args = parser.parse_args()

    x_df = pd.read_csv(args.features)
    y_df = pd.read_csv(args.labels)

    if "job_id" not in x_df.columns or "job_id" not in y_df.columns:
        raise ValueError("Both features and labels CSV must have a 'job_id' column")
    if args.target_col not in y_df.columns:
        raise ValueError(f"Target column '{args.target_col}' not in labels CSV")

    merged = x_df.merge(y_df[["job_id", args.target_col]], on="job_id", how="inner")
    if merged.empty:
        raise ValueError("No matching job_id rows found between features and labels")

    feature_cols = [c for c in merged.columns if c not in {"job_id", args.target_col}]
    if not feature_cols:
        raise ValueError("No feature columns found after merge")

    x = merged[feature_cols]
    y = merged[args.target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "target_col": args.target_col,
        },
        out,
    )

    print(f"Saved model: {out}")
    print(f"Test MAE (MHz): {mae:.2f}")
    print("Feature importances:")
    for c, imp in sorted(
        zip(feature_cols, model.feature_importances_), key=lambda t: t[1], reverse=True
    ):
        print(f"  {c}: {imp:.4f}")


if __name__ == "__main__":
    main()
