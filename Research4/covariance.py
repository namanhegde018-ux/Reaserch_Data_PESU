import pandas as pd
import os

# --------------------------------------------------
# INPUT / OUTPUT
# --------------------------------------------------

input_file = "mill_corrected.csv"
output_folder = "covariance"

os.makedirs(output_folder, exist_ok=True)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

VB_THRESHOLD = 0.30

sensor_columns = [
    "smcAC",
    "vib_table",
    "AE_table"
]

# --------------------------------------------------
# READ DATA
# --------------------------------------------------

df = pd.read_csv(input_file)

# Remove missing values
df = df.dropna(
    subset=["VB"] + sensor_columns
)

# --------------------------------------------------
# SELECT HEALTHY DATA
# --------------------------------------------------

healthy_data = df[
    df["VB"] <= VB_THRESHOLD
]

print("Total observations:", len(df))
print("Healthy observations (VB <= 0.30):", len(healthy_data))

# --------------------------------------------------
# CALCULATE OVERALL COVARIANCE
# --------------------------------------------------

covariance_matrix = healthy_data[
    sensor_columns
].cov()

# --------------------------------------------------
# DISPLAY
# --------------------------------------------------

print("\nOverall covariance matrix:")
print(covariance_matrix)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

output_file = os.path.join(
    output_folder,
    "healthy_covariance.csv"
)

covariance_matrix.to_csv(output_file)

print(f"\nSaved to: {output_file}")