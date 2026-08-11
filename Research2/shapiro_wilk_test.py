import pandas as pd
from scipy.stats import shapiro

df = pd.read_csv("mill.csv")

columns = ["AE_table", "vib_table", "smcAC"]

results = []

for case, group in df.groupby("case"):
    for col in columns:
        data = group[col].dropna()

        if len(data) >= 3:
            W, p = shapiro(data)

            results.append({
                "case": case,
                "variable": col,
                "W_statistic": W,
                "p_value": p,
                "normal": "Yes" if p > 0.05 else "No"
            })

results_df = pd.DataFrame(results)

print(results_df)

results_df.to_csv("shapiro_wilk_results.csv", index=False)