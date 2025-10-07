import pandas as pd

# Load the dataset (replace 'dataset.csv' with your file path)
df = pd.read_csv('messages_dataset.csv')

# Define the mapping from old labels to new labels
label_mapping = {
    0: 1,  # Sadness -> Low Mood
    1: 0,  # Joy -> Neutral
    2: 0,  # Love -> Neutral
    3: 3,  # Anger -> Stressed
    4: 2,  # Fear -> Anxious
    5: 0   # Surprise -> Neutral
}

# Relabel the dataset (replace 'label' with the actual column name in your dataset)
df['label'] = df['label'].map(label_mapping)

# Save the relabeled dataset to a new CSV file
df.to_csv('relabeled_messages_dataset.csv', index=False)

print("Relabeling complete. Saved to 'relabeled_messages_dataset.csv'.")