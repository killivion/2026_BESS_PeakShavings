"""Run the BESS peak-shaving forecasting pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.bess_forecasting import (
    add_german_holiday_flag,
    initialize_case,
    peak_summary,
    plot_eda,
    plot_dispatch_simulation,
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

    # GBM quantile is the recommended peak-protection model, so its dispatch
    # is the primary/unsuffixed one; conservative q80 is kept as an
    # explicitly-labeled comparison, not a strictly worse alternative -- see
    # the notebook's dispatch-policy discussion for where q80 still wins.
    simulation, scenarios = run_dispatch_demo(frame, analysis["gbm_forecast"])
    plot_dispatch_simulation(
        simulation, plot_dir, threshold_kw=80,
        forecast_label="GBM quantile forecast", filename="06_dispatch_over_time.png",
    )
    q80_simulation, q80_scenarios = run_dispatch_demo(frame, analysis["forecast"])
    plot_dispatch_simulation(
        q80_simulation, plot_dir, threshold_kw=80,
        forecast_label="conservative q80 forecast", filename="07_dispatch_over_time_q80.png",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "prepared_load.csv")
    metrics.to_csv(output_dir / "forecast_metrics.csv", index=False)
    analysis["rolling_metrics"].to_csv(output_dir / "rolling_validation_metrics.csv", index=False)
    scenarios.to_csv(output_dir / "dispatch_scenarios.csv", index=False)
    simulation.to_csv(output_dir / "dispatch_simulation.csv")
    q80_scenarios.to_csv(output_dir / "dispatch_scenarios_q80.csv", index=False)
    q80_simulation.to_csv(output_dir / "dispatch_simulation_q80.csv")

    summary = peak_summary(frame)
    summary["input_path"] = str(data_path)
    summary["plot_directory"] = str(plot_dir)
    summary["dispatch_peak_after_kw"] = float(simulation["net_load_kw"].max())
    summary["dispatch_energy_discharged_kwh"] = float(simulation["energy_discharged_kwh"].sum())
    summary["dispatch_peak_after_kw_q80"] = float(q80_simulation["net_load_kw"].max())
    summary["dispatch_energy_discharged_kwh_q80"] = float(q80_simulation["energy_discharged_kwh"].sum())

    pd.Series(summary, name="value").to_csv(output_dir / "eda_summary.csv")
    print("EDA summary")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nHoldout metrics")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nDispatch scenarios (GBM quantile, recommended)")
    print(scenarios.to_string(index=False))
    print("\nDispatch scenarios (conservative q80, comparison)")
    print(q80_scenarios.to_string(index=False))
    print(f"\nArtifacts written to {output_dir}")


if __name__ == "__main__":
    main()
