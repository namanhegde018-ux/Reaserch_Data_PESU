import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Read mill.csv
# --------------------------------------------------

mill = pd.read_csv("/home/naman-hegde/Desktop/Programs/PYTHON/Reaserch_Data_PESU/mill.csv")

# Columns required from each case file
selected_columns = [
    "Unnamed: 0",
    "smcAC",
    "vib_table",
    "AE_table"
]

# Store means and covariance matrices
case_means = {}
case_covs = {}

# Store Mahalanobis distances
all_mahalanobis = []


# --------------------------------------------------
# 2. Process case1.csv to case16.csv
# --------------------------------------------------

for i in range(1, 17):

    case_filename = f"/home/naman-hegde/Desktop/Programs/PYTHON/Reaserch_Data_PESU/Cases_CSV_files/case{i}.csv"
    selected_filename = f"case{i}_selected.csv"

    # Read case file
    case = pd.read_csv(case_filename)

    # Select required columns
    case_selected = case[selected_columns].copy()

    # Save selected columns
    case_selected.to_csv(selected_filename, index=False)

    # --------------------------------------------------
    # 3. Take 30% of rows
    # --------------------------------------------------

    n = len(case_selected)
    n_30 = int(n * 0.30)

    sample = case_selected.iloc[:n_30]

    # --------------------------------------------------
    # 4. Calculate mean and covariance
    # --------------------------------------------------

    if n_30 < 2:
        # Cannot calculate meaningful covariance
        case_means[f"case{i}_mean"] = None
        case_covs[f"case{i}_cov"] = None

        # All distances for this case = 0
        distances = np.zeros(n)

    else:

        # Use only the actual numerical measurements
        X = sample[["smcAC", "vib_table", "AE_table"]].to_numpy(
            dtype=float
        )

        # Mean vector
        mean = np.mean(X, axis=0)

        # Covariance matrix
        cov = np.cov(X, rowvar=False)

        case_means[f"case{i}_mean"] = mean
        case_covs[f"case{i}_cov"] = cov

        # --------------------------------------------------
        # 5. Calculate inverse covariance matrix
        # --------------------------------------------------

        try:
            cov_inv = np.linalg.inv(cov)

            # Check whether inverse contains valid numbers
            if not np.all(np.isfinite(cov_inv)):
                raise np.linalg.LinAlgError

            # --------------------------------------------------
            # 6. Calculate Mahalanobis distance squared
            # --------------------------------------------------

            X_all = case_selected[
                ["smcAC", "vib_table", "AE_table"]
            ].to_numpy(dtype=float)

            diff = X_all - mean

            distances = np.einsum(
                "ij,jk,ik->i",
                diff,
                cov_inv,
                diff
            )

        except (np.linalg.LinAlgError, ValueError):
            # Singular / invalid covariance matrix
            distances = np.zeros(n)


    # --------------------------------------------------
    # 7. Append this case's distances sequentially
    # --------------------------------------------------

    all_mahalanobis.extend(distances)


# --------------------------------------------------
# 8. Append Mahalanobis distance squared to mill.csv
# --------------------------------------------------

if len(all_mahalanobis) != len(mill):
    raise ValueError(
        f"Number of Mahalanobis distances ({len(all_mahalanobis)}) "
        f"does not match number of rows in mill.csv ({len(mill)})."
    )

mill["mahalanobis_distance_squared"] = all_mahalanobis


# --------------------------------------------------
# 9. Save updated mill.csv
# --------------------------------------------------

mill.to_csv("mill.csv", index=False)

print("Processing completed successfully.")

# Optional: print means and covariance matrices
for i in range(1, 17):
    print(f"\ncase{i}_mean:")
    print(case_means[f"case{i}_mean"])

    print(f"case{i}_cov:")
    print(case_covs[f"case{i}_cov"])