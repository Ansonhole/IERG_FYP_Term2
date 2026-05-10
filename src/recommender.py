from dotenv import load_dotenv
import os
import pandas as pd
from datetime import datetime
import csv
import time
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

from src.retriever import retrieve_top_k
from src.user_profile import UserProfile

def log_feedback(user_query: str, user_profile, recommended_ids: list, feedback: str = "web_auto"):
    log_file = "data/user_feedback_log.csv"
    os.makedirs("data", exist_ok=True)
    row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'user_query': user_query,
        'recommended_ids': ', '.join(map(str, recommended_ids)),
        'feedback': feedback
    }
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def generate_recommendation(user_query: str, user_profile=None, top_k: int = 6):
    try:
        # === 判斷是否為推薦意圖 ===
        recommend_keywords = ["推薦", "suggest", "最適合", "適合我", "找工作", "找實習", "recommend", "best for me", "學院推薦"]
        is_recommendation_intent = any(kw in user_query.lower() for kw in recommend_keywords)

        retrieved = retrieve_top_k(user_query, user_profile, top_k=top_k)
        recommended_ids = retrieved['id'].astype(str).tolist() if not retrieved.empty else []

        print("\n" + "="*85)
        print("🎯 CUHK 工程系個人化推薦系統")
        print("="*85 + "\n")

        # ==================== Prompt 選擇 ====================
        is_college_mode = isinstance(user_profile, dict) and user_profile.get("mode") == "college"

        if is_college_mode:
            context = retrieved.to_string(index=False)
            
            if is_recommendation_intent:
                prompt = f"""你是一位嚴謹的 CUHK 學院選擇顧問，**只能根據以下提供的真實資料**回答。

學生偏好：
1. 住宿舍：{user_profile.get('residential')}
2. 基督教價值觀：{user_profile.get('christian')}
3. 通識教育：{user_profile.get('ge_interest')}
4. 學院活動：{user_profile.get('activity')}
5. 學術/生活：{user_profile.get('academic')}
6. 性格：{user_profile.get('personality')}

真實資料：
{context}

用戶問題：{user_query}

請根據以上資料推薦學院並詳細解釋。"""
            
            else:
                # === 一般問題：直接回答，不強制 grounding ===
                prompt = f"""你是一位嚴謹的 CUHK 學院選擇顧問，**只能根據以下提供的真實資料**回答。

學生偏好：
1. 住宿舍：{user_profile.get('residential')}
2. 基督教價值觀：{user_profile.get('christian')}
3. 通識教育：{user_profile.get('ge_interest')}
4. 學院活動：{user_profile.get('activity')}
5. 學術/生活：{user_profile.get('academic')}
6. 性格：{user_profile.get('personality')}

真實資料：
{context}

用戶問題：{user_query}

請根據以上資料回答並詳細解釋。"""
            
        else:
            context = retrieved.to_string(index=False)
            if is_recommendation_intent:
                prompt = f"""你是一位嚴謹的 CUHK 工程系職業顧問，**只能根據以下提供的真實職位資料**回答。

學生背景：
- 年級：{getattr(user_profile, 'year', user_profile.get('year', '未知'))}
- 技能：{getattr(user_profile, 'skills', user_profile.get('skills', []))}
- 興趣：{getattr(user_profile, 'interests', user_profile.get('interests', ''))}
- GPA：{getattr(user_profile, 'gpa', user_profile.get('gpa', ''))}

真實職位資料：
{context}

用戶問題：{user_query}

請根據以上資料推薦適合的職位，並清楚說明原因。"""
            else:
                prompt = f"""你是一位嚴謹的 CUHK 工程系職業顧問，**只能根據以下提供的真實職位資料**回答。

學生背景：
- 年級：{getattr(user_profile, 'year', user_profile.get('year', '未知'))}
- 技能：{getattr(user_profile, 'skills', user_profile.get('skills', []))}
- 興趣：{getattr(user_profile, 'interests', user_profile.get('interests', ''))}
- GPA：{getattr(user_profile, 'gpa', user_profile.get('gpa', ''))}

真實職位資料：
{context}

用戶問題：{user_query}

請根據以上資料回答並清楚說明原因。"""
                

        # ==================== Groq 呼叫 ====================
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )

        answer = response.choices[0].message.content.strip()

        print("💡 推薦分析與建議：")
        print(answer)
        print("\n" + "="*85)

        log_feedback(user_query, user_profile, recommended_ids, "web_auto")

        return answer, retrieved

    except Exception as e:
        print("錯誤:", str(e))
        return "抱歉，系統目前發生錯誤，請稍後再試。", pd.DataFrame()
