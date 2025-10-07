import praw
import pandas as pd
import time
import os

# 🔐 Replace these with your actual credentials
reddit = praw.Reddit(
    client_id='0Guf42Z-KLLIrd2C9r-6Dw',             # <-- from the Reddit app (gray text under app name)
    client_secret='ZcTmviqBAWxkt39KzWrlI_RTebQ2ow',     # <-- from the app page
    user_agent='me by u/Hrishikesh_245'  # <-- Reddit username
)

# 🏷️ Subreddits categorized by label
label_to_subreddits = {
    0: ["adultcasualconvo"],  # Neutral
    1: [],  # Low mood
    2: [],  # Anxious
    3: []  # Stressed
}

# 📁 Output folder
output_dir = "scraped_posts"

# 📥 Post limit per subreddit (tweak to control total size)
limit = 300

# 🔁 Loop through each label and its subreddits
for label, subreddits in label_to_subreddits.items():
    for subreddit_name in subreddits:
        posts = []
        print(f"🔍 Scraping r/{subreddit_name} for label {label}...")

        try:
            subreddit = reddit.subreddit(subreddit_name)
            post_id = 1
            for submission in subreddit.hot(limit=limit):
                if not submission.stickied and submission.selftext not in ['', '[removed]', '[deleted]']:
                    posts.append({
                        'id': post_id,
                        'label': label,
                        'body': submission.selftext
                    })
                    post_id += 1

            # 💾 Save as CSV
            df = pd.DataFrame(posts)
            save_path = os.path.join(output_dir, f"{subreddit_name}_label{label}.csv")
            df.to_csv(save_path, index=False)
            print(f"✅ Saved {len(df)} posts from r/{subreddit_name} to {save_path}")

        except Exception as e:
            print(f"⚠️ Error scraping r/{subreddit_name}: {e}")

        time.sleep(2)  # ⏳ Pause to avoid rate limits