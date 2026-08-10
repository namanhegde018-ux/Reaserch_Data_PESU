import numpy as np
import pandas as pd

from MeanCovariance import cov_matrix, mean

input_csv = "mill_selected.csv"
df = pd.read_csv(input_csv)

id_column_name = df.columns[0]

feature_matrix = df.drop(columns=[id_column_name]).to_numpy()

mean = np.array(mean)
cov = np.array(cov_matrix)

inv_cov = np.linalg.pinv(cov)

diff = feature_matrix - mean

mahalanobis_distances = np.sqrt(np.sum((diff @ inv_cov) * diff, axis=1))

output_df = pd.DataFrame(
    {
        id_column_name: df[id_column_name],
        "mahalanobis_distance": mahalanobis_distances**2,
    }
)

output_csv = "mahalanobis_results.csv"
output_df.to_csv(output_csv, index=False)

print(f"Done!")