# src/evaluator.py
import pandas as pd
from src.retriever import retrieve_top_k
from src.user_profile import UserProfile

def calculate_metrics(recommended_ids, ground_truth_ids, k=5):
    relevant = len(set(recommended_ids[:k]) & set(ground_truth_ids))
    precision_k = relevant / k
    recall_k = relevant / len(ground_truth_ids) if ground_truth_ids else 0
    
    # NDCG
    y_true = [1 if str(id) in [str(x) for x in ground_truth_ids] else 0 for id in recommended_ids[:k]]
    y_score = [1.0 / (i+1) for i in range(k)]
    from sklearn.metrics import ndcg_score
    ndcg = ndcg_score([y_true], [y_score], k=k)
    
    return {
        "Precision@K": round(precision_k, 4),
        "Recall@K": round(recall_k, 4),
        "NDCG@K": round(ndcg, 4)
    }

def run_full_evaluation():
    print("\n" + "="*80)
    print("📊 Data Science 推薦系統完整離線評估")
    print("="*80)

    # Test Case 1: AI 相關
    user1 = UserProfile(year="3", skills=["python", "machine learning", "ai"], interests="AI", gpa=3.7)
    results1 = retrieve_top_k("推薦 AI 相關實習或畢業職位", user1, top_k=5)
    rec_ids1 = results1['id'].astype(str).tolist()
    print(f"Test 1 - AI 推薦 ID: {rec_ids1}")

    # Test Case 2: 軟體工程
    user2 = UserProfile(year="2", skills=["java", "react", "sql"], interests="web", gpa=3.4)
    results2 = retrieve_top_k("推薦軟體工程實習", user2, top_k=5)
    rec_ids2 = results2['id'].astype(str).tolist()
    print(f"Test 2 - 軟體工程推薦 ID: {rec_ids2}")

    # 這裡你可以手動填入 ground truth（你認為正確的 ID）
    # 例如：
    # ground_truth1 = ['15', '2', '52']
    # metrics1 = calculate_metrics(rec_ids1, ground_truth1)

    print("\n✅ 評估完成！你可以繼續擴充 ground truth 來計算精確指標。")