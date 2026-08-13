import pandas as pd

df = pd.read_csv("mill_corrected.csv")

healthy_data = df[df["VB"] <= 0.30]

print("Total observations:", len(df))
print("Healthy observations:", len(healthy_data))
print("Unhealthy observations:", len(df) - len(healthy_data))


# (0.99*(87-1)) + 1;