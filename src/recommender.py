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
        retrieved = retrieve_top_k(user_query, user_profile, top_k=top_k)
        recommended_ids = retrieved['id'].astype(str).tolist() if not retrieved.empty else []

        print("\n" + "="*85)
        print("🎯 CUHK 工程系個人化推薦系統")
        print("="*85 + "\n")

        context_parts = []
        is_college_mode = isinstance(user_profile, dict) and user_profile.get("mode") == "college"

        for i, (_, row) in enumerate(retrieved.iterrows(), 1):
            score = row.get('final_score', row.get('similarity_score', 0))
            if row.get('type') == "college_choice":
                print(f"【{i}】 學院：{row.get('college')} | 分數: {score:.3f}")
                print(f"問題：{row.get('question')}")
                print(f"回答：{str(row.get('answer', ''))[:380]}...\n")
                context_parts.append(f"學院：{row.get('college')}\nQ: {row.get('question')}\nA: {row.get('answer')}")
            else:
                print(f"【{i}】 ID: {row.get('id')} | 分數: {score:.3f}")
                print(f"職位：{row.get('job_title', row.get('title', '未提供'))}")
                print(f"公司：{row.get('company_name', row.get('company', '未提供'))}")
                print(f"截止日期：{row.get('deadline', '未提供')}")
                print("-" * 70)
                context_parts.append(f"職位：{row.get('job_title', row.get('title', ''))} at {row.get('company_name', '')}")

        context = "\n\n".join(context_parts)

        # Prompt
        if is_college_mode:
            prompt = f"""你是一位專業且有經驗的 CUHK 學院選擇顧問。
學生回答：
1. 住宿舍意願：{user_profile.get('residential')}
2. 基督教價值觀：{user_profile.get('christian')}
3. 通識教育興趣：{user_profile.get('ge_interest')}
4. 參與學院活動：{user_profile.get('activity')}
5. 學術 vs 生活：{user_profile.get('academic')}
6. 性格傾向：{user_profile.get('personality')}

用戶問題：{user_query}

請根據以上答案，推薦1-2個最適合的學院，並詳細解釋原因。語氣親切，像老師在給個人建議。"""
        else:
            prompt = f"""你是一位專業親切的 CUHK 工程學院升學就業顧問。
用戶問題：{user_query}
參考資料：
{context}
請用自然、清楚、有幫助的方式回答。"""

        # Groq 呼叫 + 重試
        answer = "目前系統忙碌中，請稍後再試。"
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1024
                )
                answer = response.choices[0].message.content.strip()
                break
            except Exception as e:
                if attempt == 4:
                    print(f"Groq 最終失敗: {e}")
                else:
                    wait = 2 ** attempt
                    print(f"Groq 錯誤，重試中... ({attempt+1}/5)")
                    time.sleep(wait)

        print("💡 推薦分析與建議：")
        print(answer)
        print("\n" + "="*85)

        log_feedback(user_query, user_profile, recommended_ids, "web_auto")

        return answer, retrieved

    except Exception as e:
        print("generate_recommendation 錯誤:", str(e))
        return f"抱歉，系統目前發生錯誤，請稍後再試。", pd.DataFrame()
