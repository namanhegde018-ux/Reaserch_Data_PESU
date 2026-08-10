import pandas as pd

df = pd.read_csv('baseline.csv')

columns_to_keep = ['Unnamed: 0', 'smcAC', 'vib_table', 'AE_table']

filtered_df = df[columns_to_keep]

filtered_df.to_csv('baseline_selected.csv', index=False)

print("Done!")