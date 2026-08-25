"""Run the BESS peak-shaving forecasting pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.bess_forecasting import (
    add_german_holiday_flag,
    initialize_case,
    peak_summary,
    plot_eda,
    run_dispatch_demo,
    run_forecast_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BESS load forecasting case study")
    parser.add_argument("--input", default="load_timeseries_2025_case_study.csv")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    root = Path.cwd()
    requested_input = input_path if input_path.is_absolute() else root / input_path
    if not requested_input.exists() and input_path.parent == Path("."):
        requested_input = root / "case_inputs" / input_path.name
    data_path, _, plot_dir, frame, _ = initialize_case(root, requested_input)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    plot_dir = output_dir / "plots"
    plot_eda(frame, plot_dir)
    analysis = run_forecast_analysis(frame, plot_dir)
    metrics = analysis["metrics"]
    frame = add_german_holiday_flag(frame)
    simulation, scenarios = run_dispatch_demo(frame, analysis["forecast"])
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "prepared_load.csv")
    metrics.to_csv(output_dir / "forecast_metrics.csv", index=False)
    scenarios.to_csv(output_dir / "dispatch_scenarios.csv", index=False)
    simulation.to_csv(output_dir / "dispatch_simulation.csv")
    summary = peak_summary(frame)
    summary["input_path"] = str(data_path)
    summary["plot_directory"] = str(plot_dir)
    summary["dispatch_peak_after_kw"] = float(simulation["net_load_kw"].max())
    summary["dispatch_energy_discharged_kwh"] = float(simulation["energy_discharged_kwh"].sum())
    import pandas as pd

    pd.Series(summary, name="value").to_csv(output_dir / "eda_summary.csv")
    print("EDA summary")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nHoldout metrics")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nDispatch scenarios")
    print(scenarios.to_string(index=False))
    print(f"\nArtifacts written to {output_dir}")


if __name__ == "__main__":
    main()
