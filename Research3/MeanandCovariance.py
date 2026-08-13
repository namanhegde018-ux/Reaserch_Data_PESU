import pandas as pd
import numpy as np
import os

input_file = "mill_corrected.csv"
output_folder = "healthy_baseline"

os.makedirs(output_folder, exist_ok=True)

HEALTHY_VB_THRESHOLD = 0.30

sensor_columns = [
    "smcAC",
    "vib_table",
    "AE_table"
]

df = pd.read_csv(input_file)

df = df.dropna(
    subset=["case", "VB"] + sensor_columns
)

summary = []

for case_number in sorted(df["case"].unique()):

    # Select current case
    case_data = df[df["case"] == case_number].copy()

    # Select healthy observations
    healthy_data = case_data[
        case_data["VB"] <= HEALTHY_VB_THRESHOLD
    ].copy()

    # Check whether enough healthy data exists
    if len(healthy_data) < 2:
        print(
            f"Case {case_number}: "
            f"Not enough healthy observations"
        )
        continue


    mean_values = healthy_data[sensor_columns].mean()

    covariance_matrix = healthy_data[
        sensor_columns
    ].cov()

    mean_df = pd.DataFrame({
        "sensor": sensor_columns,
        "healthy_mean": [
            mean_values[col] for col in sensor_columns
        ]
    })

    mean_file = os.path.join(
        output_folder,
        f"case_{int(case_number)}_healthy_mean.csv"
    )

    mean_df.to_csv(mean_file, index=False)

    covariance_file = os.path.join(
        output_folder,
        f"case_{int(case_number)}_healthy_covariance.csv"
    )

    covariance_matrix.to_csv(covariance_file)

    summary.append({
        "case": case_number,
        "total_observations": len(case_data),
        "healthy_observations": len(healthy_data),
        "minimum_VB_healthy": healthy_data["VB"].min(),
        "maximum_VB_healthy": healthy_data["VB"].max(),
        "mean_smcAC": mean_values["smcAC"],
        "mean_vib_table": mean_values["vib_table"],
        "mean_AE_table": mean_values["AE_table"]
    })


summary_df = pd.DataFrame(summary)

summary_file = os.path.join(
    output_folder,
    "healthy_baseline_summary.csv"
)

summary_df.to_csv(summary_file, index=False)

print("\nHealthy baseline calculation completed.")
print(f"Output folder: {output_folder}")
print("\nSummary:")
print(summary_df)