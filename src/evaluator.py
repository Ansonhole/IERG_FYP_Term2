import pandas as pd
import os
from datetime import datetime
import csv
import numpy as np

def calculate_metrics(recommended_ids, ground_truth_ids, k=5):
    """計算推薦系統評估指標（安全處理 NaN 和邊界情況）"""
    if not recommended_ids or not ground_truth_ids:
        return {
            "Precision@K": 0.0,
            "Recall@K": 0.0,
            "NDCG@K": 0.0,
            "Hits": 0,
            "K": k
        }
    
    recommended_set = set(str(id) for id in recommended_ids[:k])
    ground_truth_set = set(str(id) for id in ground_truth_ids)
    
    relevant = len(recommended_set & ground_truth_set)
    
    precision_k = relevant / k if k > 0 else 0.0
    recall_k = relevant / len(ground_truth_set) if ground_truth_set else 0.0
    
    # NDCG 安全計算
    try:
        y_true = [1 if str(rid) in ground_truth_set else 0 for rid in recommended_ids[:k]]
        from sklearn.metrics import ndcg_score
        ndcg = ndcg_score([y_true], [[1.0] * len(y_true)], k=k)
        if np.isnan(ndcg) or np.isinf(ndcg):
            ndcg = 0.0
    except:
        ndcg = 0.0
    
    return {
        "Precision@K": round(float(precision_k), 4),
        "Recall@K": round(float(recall_k), 4),
        "NDCG@K": round(float(ndcg), 4),
        "Hits": relevant,
        "K": k
    }


def save_evaluation_log(user_query, recommended_ids, ground_truth_ids, metrics):
    """儲存評估記錄"""
    log_file = "data/user_feedback_log.csv"
    os.makedirs("data", exist_ok=True)
    
    row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'user_query': user_query,
        'recommended_ids': ','.join(map(str, recommended_ids)),
        'ground_truth_ids': ','.join(map(str, ground_truth_ids)),
        'precision': metrics["Precision@K"],
        'recall': metrics["Recall@K"],
        'ndcg': metrics["NDCG@K"]
    }
    
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
