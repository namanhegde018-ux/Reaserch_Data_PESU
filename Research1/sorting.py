import pandas as pd

INPUT_FILE = 'mill.csv'
TARGET_COLUMN = 'case'

for i in range (1,17):
    OUTPUT_FILE = f"case{i}.csv"
    TARGET_VALUE = f"{i}"
    df = pd.read_csv(INPUT_FILE)
    filtered_df = df[df[TARGET_COLUMN].astype(str).str.strip() == str(TARGET_VALUE).strip()]
    filtered_df.to_csv(OUTPUT_FILE, index=False)

print("Done!")