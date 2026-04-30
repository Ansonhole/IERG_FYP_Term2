import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.data_loader import load_all_data
import os

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index("data/faiss_index.bin")
df = pd.read_pickle("data/metadata.pkl")

# 載入 popularity 和 collaborative filtering 分數
def load_popularity_scores():
    log_file = "data/user_feedback_log.csv"
    if not os.path.exists(log_file):
        return pd.Series(dtype=float)
    try:
        log = pd.read_csv(log_file)
        popularity = log['recommended_ids'].str.split(', ').explode().value_counts()
        return popularity
    except:
        return pd.Series(dtype=float)

popularity_scores = load_popularity_scores()

def load_cf_scores():
    log_file = "data/user_feedback_log.csv"
    if not os.path.exists(log_file):
        return pd.Series(dtype=float)
    try:
        log = pd.read_csv(log_file)
        cf = log['recommended_ids'].str.split(', ').explode().value_counts()
        return cf
    except:
        return pd.Series(dtype=float)

cf_scores = load_cf_scores()

def normalize_skill(skill_text):
    if pd.isna(skill_text):
        return []
    if isinstance(skill_text, list):
        return [str(s).lower().strip() for s in skill_text]
    if isinstance(skill_text, str):
        return [s.lower().strip() for s in skill_text.split(',')]
    return []

def skill_match_score(row, user_profile):
    if user_profile is None or not user_profile.skills:
        return 0.0
    user_skills = [s.lower().strip() for s in user_profile.skills]
    required = normalize_skill(row.get('required_skills', ''))
    preferred = normalize_skill(row.get('preferred_skills', ''))
    
    req_match = sum(1 for s in required if any(u in s or s in u for u in user_skills)) / max(len(required), 1)
    pref_match = sum(1 for s in preferred if any(u in s or s in u for u in user_skills)) * 0.6
    return req_match * 0.7 + pref_match * 0.3

def mmr_rerank(results, query_emb, lambda_param=0.5, top_k=6):
    """Maximum Marginal Relevance - 增加推薦多樣性"""
    if len(results) <= top_k:
        return results
    
    selected = []
    candidates = results.copy()
    
    while len(selected) < top_k and len(candidates) > 0:
        scores = []
        for idx, row in candidates.iterrows():
            sim = row['similarity_score']
            if len(selected) == 0:
                score = sim
            else:
                # 計算與已選結果的最大相似度
                max_sim = max([s['similarity_score'] for s in selected])
                diversity = 1 - max_sim
                score = lambda_param * sim + (1 - lambda_param) * diversity
            scores.append(score)
        
        best_idx = candidates.index[np.argmax(scores)]
        selected.append(candidates.loc[best_idx])
        candidates = candidates.drop(best_idx)
    
    return pd.DataFrame(selected)

def retrieve_top_k(query: str, user_profile=None, top_k: int = 6):
    query_emb = model.encode([query])[0].astype('float32')
    faiss.normalize_L2(query_emb.reshape(1, -1))
    
    distances, indices = index.search(query_emb.reshape(1, -1), min(top_k * 5, len(df)))
    results = df.iloc[indices[0]].copy()
    results["similarity_score"] = distances[0]

    # Skill Match
    if user_profile and not isinstance(user_profile, dict):
        results["skill_score"] = results.apply(lambda row: skill_match_score(row, user_profile), axis=1)
    else:
        results["skill_score"] = 0.0

    # Popularity
    results["popularity_score"] = results['id'].astype(str).map(popularity_scores).fillna(0)
    results["popularity_score"] = results["popularity_score"] / (results["popularity_score"].max() + 1e-8)

    # Collaborative Filtering
    results["cf_score"] = results['id'].astype(str).map(cf_scores).fillna(0)
    results["cf_score"] = results["cf_score"] / (results["cf_score"].max() + 1e-8)

    # 最終分數
    if user_profile and not isinstance(user_profile, dict):
        results["final_score"] = (
            results["similarity_score"] * 0.45 +
            results["skill_score"] * 0.25 +
            results["popularity_score"] * 0.15 +
            results["cf_score"] * 0.15
        )
    else:
        results["final_score"] = results["similarity_score"]

    # 先排序再做 MMR
    results = results.sort_values(by="final_score", ascending=False).head(top_k * 2)
    
    # MMR 多樣性重排序
    results = mmr_rerank(results, query_emb, lambda_param=0.5, top_k=top_k)

    return results