import pandas as pd

# Load your CSV file
df = pd.read_csv("resolved_partial_1.csv")

# Make sure the ID column is named correctly. Replace 'id' with your actual column name if different.
existing_ids = set(df['id'].astype(int))

# Full range of expected IDs
expected_ids = set(range(1, 17224))

# Find missing IDs
missing_ids = expected_ids - existing_ids

# Print missing IDs
if missing_ids:
    print("Missing IDs:")
    for mid in sorted(missing_ids):
        print(mid)
else:
    print("All IDs from 1 to 17223 are present.")