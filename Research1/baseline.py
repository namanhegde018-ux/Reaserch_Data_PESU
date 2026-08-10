import pandas as pd
import os

OUTPUT_FILE = 'baseline.csv'

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

for i in range(1, 16):
    INPUT_FILE = f"case{i}.csv"

    if os.path.exists(INPUT_FILE):
        df = pd.read_csv(INPUT_FILE)
        num_rows_to_keep = int(len(df) * 0.30)
        df_30_percent = df.head(num_rows_to_keep).copy()
        df_30_percent['Source_Case'] = f"Case_{i}"
        file_exists = os.path.exists(OUTPUT_FILE)
        df_30_percent.to_csv(OUTPUT_FILE, mode='a', header=not file_exists, index=False)

print("Done!")