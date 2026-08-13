import pandas as pd
from scipy.stats import shapiro

df = pd.read_csv("mill.csv")

columns = ["AE_table", "vib_table", "smcAC"]

for col in columns:
    data = df[col].dropna()

    W, p = shapiro(data)

    print(f"{col}:")
    print(f"W = {W:.6f}")
    print(f"p-value = {p:.6f}")

    if p > 0.05:
        print("Overall: Normally distributed")
    else:
        print("Overall: Not normally distributed")

    print()