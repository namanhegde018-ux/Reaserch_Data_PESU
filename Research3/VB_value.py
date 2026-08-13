import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/home/naman-hegde/Desktop/Programs/PYTHON/Reaserch_Data_PESU/mill.csv")

column = "VB"

missing_rows = df[df[column].isna()].index

print("Missing rows:")
print(missing_rows)

plt.figure(figsize=(12, 5))

plt.plot(
    df.index,
    df[column],
    marker=".",
    linestyle="-",
    label=column
)

plt.scatter(
    missing_rows,
    [0] * len(missing_rows),
    marker="x",
    label="Missing values"
)

plt.xlabel("Row number")
plt.ylabel(column)
plt.title(f"{column} vs Row Number")
plt.grid(True)
plt.legend()

plt.show()

df[column] = df[column].interpolate(
    method="linear"
)

print("\nRemaining missing values:")
print(df[column].isna().sum())

plt.figure(figsize=(12, 5))

plt.plot(
    df.index,
    df[column],
    marker=".",
    linestyle="-"
)

plt.xlabel("Row number")
plt.ylabel(column)
plt.title(f"{column} vs Row Number — After Interpolation")
plt.grid(True)

plt.show()

df.to_csv("mill_corrected.csv", index=False)