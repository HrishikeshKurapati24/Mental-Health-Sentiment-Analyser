import pandas as pd

# Path to your CSV file
file_path = 'full_reddit_dataset_1.csv'

# Read the CSV
df = pd.read_csv(file_path)

# Check if 'label' column exists
if 'label' not in df.columns:
    raise ValueError("The CSV file must contain a 'label' column.")

# Count occurrences of each label
label_counts = df['label'].value_counts().sort_index()

# Print counts for 0, 1, 2, 3
for label in range(4):
    count = label_counts.get(label, 0)
    print(f"Label {label}: {count} occurrences")