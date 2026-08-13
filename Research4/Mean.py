import pandas as pd
import os

# --------------------------------------------------
# INPUT / OUTPUT
# --------------------------------------------------

input_file = "mill_corrected.csv"
output_folder = "healthy_baseline"

os.makedirs(output_folder, exist_ok=True)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

HEALTHY_VB_THRESHOLD = 0.30

sensor_columns = [
    "smcAC",
    "vib_table",
    "AE_table"
]

# --------------------------------------------------
# READ DATA
# --------------------------------------------------

df = pd.read_csv(input_file)

# Remove rows with missing required values
df = df.dropna(
    subset=["case", "VB"] + sensor_columns
)

# --------------------------------------------------
# CALCULATE MEAN FOR EACH CASE
# --------------------------------------------------

summary = []

for case_number in sorted(df["case"].unique()):

    # Select current case
    case_data = df[df["case"] == case_number]

    # Select healthy observations
    healthy_data = case_data[
        case_data["VB"] <= HEALTHY_VB_THRESHOLD
    ]

    # Check if healthy data exists
    if len(healthy_data) == 0:
        print(f"Case {case_number}: No healthy observations")
        continue

    # Calculate mean
    mean_values = healthy_data[sensor_columns].mean()

    summary.append({
        "case": case_number,
        "healthy_observations": len(healthy_data),
        "mean_smcAC": mean_values["smcAC"],
        "mean_vib_table": mean_values["vib_table"],
        "mean_AE_table": mean_values["AE_table"]
    })

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

summary_df = pd.DataFrame(summary)

output_file = os.path.join(
    output_folder,
    "case_wise_healthy_means.csv"
)

summary_df.to_csv(output_file, index=False)

# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print("\nCase-wise healthy means:")
print(summary_df)

print(f"\nSaved to: {output_file}")