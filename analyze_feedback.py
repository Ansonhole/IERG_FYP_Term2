import pandas as pd
import os
from datetime import datetime

print("=== Feedback Analysis Tool ===\n")

log_file = "data/user_feedback_log.csv"

if not os.path.exists(log_file):
    print("❌ No feedback log found yet. Please use the system first to collect feedback.")
    exit()

df = pd.read_csv(log_file)

print(f"Total feedback collected: {len(df)} entries\n")

# Basic Statistics
print("=== Basic Statistics ===")
print(f"Total unique users (by skills): {df['user_skills'].nunique()}")
print(f"Average number of liked jobs per query: {df['feedback'].apply(lambda x: len(str(x).split(',')) if x != 'skip' else 0).mean():.2f}\n")

# Most liked job IDs
liked_jobs = df['feedback'].dropna().str.split(',').explode().str.strip()
liked_jobs = liked_jobs[liked_jobs != '']
print("=== Most Liked Job IDs ===")
print(liked_jobs.value_counts().head(10))

# Most common skills in liked jobs
print("\n=== Most Common Skills in Liked Jobs ===")
skills = df['user_skills'].str.split(',').explode().str.strip()
print(skills.value_counts().head(10))

# Save summary
summary_file = "data/feedback_summary.txt"
with open(summary_file, "w", encoding="utf-8") as f:
    f.write(f"Feedback Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write("="*50 + "\n")
    f.write(f"Total feedback entries: {len(df)}\n")
    f.write(f"Most liked jobs:\n{liked_jobs.value_counts().head(10)}\n")

print(f"\n✅ Analysis completed! Summary saved to 'data/feedback_summary.txt'")