import pandas as pd
from src.evaluator import evaluate_recommendation
from src.retriever import retrieve_top_k
from src.user_profile import UserProfile

print("=== Recommender System Evaluation ===\n")

# Example test case (you can modify this)
user_profile = UserProfile(
    year=3,
    skills=["python", "machine learning", "ai"],
    interests=["data science", "software engineering"],
    gpa=3.4
)

query = "推薦 AI 或 Machine Learning 相關的實習或畢業職位"

retrieved = retrieve_top_k(query, user_profile, k=5)

# Ground truth = jobs you think are good for this user
ground_truth = ["F", "G", "J", "O"]   # Change these IDs based on your data

metrics = evaluate_recommendation(retrieved, ground_truth, k=5)

print("Evaluation Results:")
for metric, value in metrics.items():
    print(f"   {metric}: {value}")

print("\n✅ Evaluation completed!")