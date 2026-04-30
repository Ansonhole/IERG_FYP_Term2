import pandas as pd
import os

print("=== Feedback-based Ranking Weight Updater ===\n")

log_file = "data/user_feedback_log.csv"

if not os.path.exists(log_file):
    print("No feedback log found yet.")
    exit()

df = pd.read_csv(log_file)

# Count how many times each job ID was liked
liked = df['feedback'].dropna().str.split(',').explode().str.strip()
liked_counts = liked.value_counts()

print("Most liked job IDs:")
print(liked_counts.head(10))

# Simple idea: Increase weight for skills that appear in liked jobs
print("\nThis data can be used to adjust skill weights in future versions.")

# For now, we just analyze. 
# In advanced version, we can train a small model to predict 'like' probability.