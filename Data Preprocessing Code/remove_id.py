import pandas as pd

# Load the CSV file
df = pd.read_csv('reddit_posts_data_2.csv')  # Replace with your CSV filename

# Remove the 'id' column if it exists
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Save the modified DataFrame to a new CSV file
df.to_csv('output.csv', index=False)