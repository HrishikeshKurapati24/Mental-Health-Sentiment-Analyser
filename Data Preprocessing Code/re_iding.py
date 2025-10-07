import pandas as pd

# Load your dataset
df = pd.read_csv('filtered_psychforums_dataset.csv')

# Relabel the ids to be unique and sequential
df['id'] = range(1, len(df) + 1)

# Save the updated dataset
df.to_csv('psychforums_multilabel.csv', index=False)