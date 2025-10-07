import pandas as pd

# File paths
source_csv = "Datasets/twitter_messages_kaggle.csv"     # File from which to sample label 3 examples
target_csv = "full_reddit_dataset.csv"     # File to which samples will be appended
output_csv = "full_reddit_dataset_1.csv"   # Final output file

# Load source and target CSVs
source_df = pd.read_csv(source_csv)
target_df = pd.read_csv(target_csv)

# Filter label 3 examples from the source file
label_3_df = source_df[source_df['label'] == 3]

# Check if enough label 3 examples are available
if len(label_3_df) < 1000:
    raise ValueError(f"Only {len(label_3_df)} label 3 examples found. Cannot sample 1000 without replacement.")

# Sample 1000 unique examples (no replacement)
sampled_label_3 = label_3_df.sample(n=1000, replace=False, random_state=42)

# Append sampled examples to target dataframe
combined_df = pd.concat([target_df, sampled_label_3], ignore_index=True)

# Save to output file
combined_df.to_csv(output_csv, index=False)

print(f"Appended 1000 label 3 examples from {source_csv} to {target_csv}. Saved as {output_csv}.")