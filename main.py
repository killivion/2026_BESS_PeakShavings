"""Run the BESS peak-shaving forecasting pipeline."""
from __future__ import annotations

import argparse

from src.bess_forecasting import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BESS load forecasting case study")
    parser.add_argument("--input", default="load_timeseries_2025_case_study.csv")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    _, metrics, summary = run_pipeline(args.input, args.output_dir)
    print("EDA summary")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nHoldout metrics")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
