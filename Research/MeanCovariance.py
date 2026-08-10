import pandas as pd
import numpy as np

df = pd.read_csv("baseline_selected.csv")

data = df.iloc[:, 1:].to_numpy()

mean = np.mean(data, axis=0).reshape(1, 3)

cov_matrix = np.cov(data, rowvar=False)

print("Mean:")
print(mean)

print("\nCovariance Matrix:")
print(cov_matrix)

print("\nShape:", cov_matrix.shape)