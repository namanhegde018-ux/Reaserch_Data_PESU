import pandas as pd
import numpy as np
import os

# --------------------------------------------------
# INPUT FILES
# --------------------------------------------------

input_file = "mill_corrected.csv"
mean_file = "healthy_baseline/case_wise_healthy_means.csv"
covariance_file = "covariance/healthy_covariance.csv"

output_file = "mahalanobis_distance.csv"

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

sensor_columns = [
    "smcAC",
    "vib_table",
    "AE_table"
]

# --------------------------------------------------
# READ DATA
# --------------------------------------------------

df = pd.read_csv(input_file)

mean_df = pd.read_csv(mean_file)

covariance_df = pd.read_csv(
    covariance_file,
    index_col=0
)

# --------------------------------------------------
# CONVERT MEANS INTO CASE LOOKUP
# --------------------------------------------------
mean_lookup = mean_df.set_index("case")[
    ["mean_smcAC", "mean_vib_table", "mean_AE_table"]
]

mean_lookup.columns = sensor_columns

# --------------------------------------------------
# COVARIANCE MATRIX
# --------------------------------------------------

covariance_matrix = covariance_df[
    sensor_columns
].loc[sensor_columns].values

# --------------------------------------------------
# INVERSE COVARIANCE
# --------------------------------------------------

covariance_inverse = np.linalg.pinv(
    covariance_matrix
)

# --------------------------------------------------
# CALCULATE MAHALANOBIS DISTANCE
# --------------------------------------------------

mahalanobis_values = []

for index, row in df.iterrows():

    case_number = row["case"]

    # Check whether mean exists for this case
    if case_number not in mean_lookup.index:
        mahalanobis_values.append(np.nan)
        continue

    # Sensor observation
    x = row[sensor_columns].values.astype(float)

    # Mean corresponding to this case
    mean = mean_lookup.loc[
        case_number
    ].values.astype(float)

    # Difference from case-specific mean
    difference = x - mean

    # Mahalanobis Distance Squared
    distance_squared = (
        difference.T
        @ covariance_inverse
        @ difference
    )

    mahalanobis_values.append(distance_squared)

# --------------------------------------------------
# ADD RESULT
# --------------------------------------------------

df["mahalanobis_distance_squared"] = mahalanobis_values

# --------------------------------------------------
# SAVE
# --------------------------------------------------

df.to_csv(output_file, index=False)

print("Mahalanobis distance calculation completed.")
print(f"Output saved to: {output_file}")

print("\nFirst few results:")
print(
    df[
        ["case", "VB"] +
        sensor_columns +
        ["mahalanobis_distance_squared"]
    ].head()
)