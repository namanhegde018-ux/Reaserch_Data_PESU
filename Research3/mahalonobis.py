import pandas as pd
import numpy as np
import os

input_file = "mill_corrected.csv"
baseline_folder = "healthy_baseline"
output_file = "mahalanobis_distance_squared.csv"

sensor_columns = [
    "smcAC",
    "vib_table",
    "AE_table"
]

df = pd.read_csv(input_file)

results = []

for case_number in sorted(df["case"].unique()):

    print(f"Processing Case {case_number}...")

    case_data = df[
        df["case"] == case_number
    ].copy()

    mean_file = os.path.join(
        baseline_folder,
        f"case_{int(case_number)}_healthy_mean.csv"
    )

    covariance_file = os.path.join(
        baseline_folder,
        f"case_{int(case_number)}_healthy_covariance.csv"
    )

    if not os.path.exists(mean_file):
        print(f"Mean file missing for Case {case_number}")
        continue

    if not os.path.exists(covariance_file):
        print(
            f"Covariance file missing for Case {case_number}"
        )
        continue

    mean_df = pd.read_csv(mean_file)

    mean_vector = mean_df[
        "healthy_mean"
    ].to_numpy()

    covariance_matrix = pd.read_csv(
        covariance_file,
        index_col=0
    ).to_numpy()

    covariance_inverse = np.linalg.pinv(
        covariance_matrix
    )

    # --------------------------------------------------
    # SENSOR DATA
    # --------------------------------------------------

    X = case_data[
        sensor_columns
    ].to_numpy()

    # --------------------------------------------------
    # DIFFERENCE FROM HEALTHY MEAN
    # --------------------------------------------------

    difference = X - mean_vector

    # --------------------------------------------------
    # MAHALANOBIS DISTANCE SQUARED
    # --------------------------------------------------

    mahalanobis_squared = np.einsum(
        "ij,jk,ik->i",
        difference,
        covariance_inverse,
        difference
    )

    # Add result
    case_data[
        "mahalanobis_distance_squared"
    ] = mahalanobis_squared

    # --------------------------------------------------
    # ADD TO RESULTS
    # --------------------------------------------------

    results.append(case_data)

# --------------------------------------------------
# COMBINE ALL CASES
# --------------------------------------------------

if results:

    final_df = pd.concat(
        results,
        ignore_index=True
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    final_df.to_csv(
        output_file,
        index=False
    )

    print("\nCalculation completed.")
    print(f"Output saved as: {output_file}")

else:

    print("No results were generated.")