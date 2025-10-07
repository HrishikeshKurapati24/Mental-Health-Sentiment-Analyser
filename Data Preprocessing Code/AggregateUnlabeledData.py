import pandas as pd

def aggregate_csvs():
    file_names = [
        "DecidingToBeBetter_posts.csv",
        "Diary_posts.csv",
        "Depression_posts.csv",
        "mentalhealth_posts.csv",
        "TrueOffMyChest_posts.csv"
    ]

    aggregated_df = pd.DataFrame(columns=["id", "text"])

    for file in file_names:
        try:
            df = pd.read_csv(file, usecols=["id", "text"])
            aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)
            print(f"Loaded {len(df)} rows from {file}")
        except Exception as e:
            print(f"Failed to load {file}: {e}")

    print(f"\nTotal aggregated examples: {len(aggregated_df)}")
    return aggregated_df

# Example usage:
aggregated_data = aggregate_csvs()
aggregated_data.to_csv("unlabeled_data.csv", index=False)