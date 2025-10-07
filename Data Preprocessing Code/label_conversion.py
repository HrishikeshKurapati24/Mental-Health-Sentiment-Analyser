import pandas as pd

df = pd.read_csv("Datasets/synthetic_mental_health_journals.csv")

# Define the mapping
label_map = {
    'Low Mood': 1,
    'Anxious/Worried': 2,
    'Stressed/Overwhelmed': 3,
    'Stable/Neutral': 0
}

# Apply the mapping
df['label'] = df['label'].map(label_map)

# Save the modified CSV
df.to_csv('synthetic_mental_health_jounrnals_dataset.csv', index=False)

print("Label mapping completed and file saved as 'updated_file.csv'.")