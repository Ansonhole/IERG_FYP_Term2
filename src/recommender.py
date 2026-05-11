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
        # === Detect recommendation intent ===
        recommend_keywords = ["推薦", "suggest", "最適合", "適合我", "找工作", "找實習", "recommend", "best for me", "學院推薦"]
        is_recommendation_intent = any(kw in user_query.lower() for kw in recommend_keywords)

        retrieved = retrieve_top_k(user_query, user_profile, top_k=top_k)
        recommended_ids = retrieved['id'].astype(str).tolist() if not retrieved.empty else []

        print("\n" + "="*85)
        print("🎯 CUHK Engineering Personalized Recommendation System")
        print("="*85 + "\n")

        is_college_mode = isinstance(user_profile, dict) and user_profile.get("mode") == "college"

        # ==================== Prompt Selection (English) ====================
        if is_college_mode:
            context = retrieved.to_string(index=False)
            
            if is_recommendation_intent:
                prompt = f"""You are a rigorous and experienced CUHK college selection advisor. You must ONLY answer based on the real data provided below. Do not make up information.

Student's Preferences:
1. Willing to live in college hostel: {user_profile.get('residential')}
2. Importance of Christian values: {user_profile.get('christian')}
3. Interest in General Education (GE): {user_profile.get('ge_interest')}
4. Willing to join college activities: {user_profile.get('activity')}
5. Academic vs Campus Life: {user_profile.get('academic')}
6. Personality: {user_profile.get('personality')}

Real College Data:
{context}

User Question: {user_query}

Please recommend the most suitable college(s) and explain in detail why they are suitable."""
            else:
                prompt = f"""You are a rigorous CUHK college advisor. Answer based on the real data provided.

Student's Preferences:
1. Willing to live in college hostel: {user_profile.get('residential')}
2. Importance of Christian values: {user_profile.get('christian')}
3. Interest in General Education: {user_profile.get('ge_interest')}
4. Willing to join college activities: {user_profile.get('activity')}
5. Academic vs Campus Life: {user_profile.get('academic')}
6. Personality: {user_profile.get('personality')}

Real Data:
{context}

User Question: {user_query}

Please answer clearly and explain based on the data."""

        else:
            # Work / Internship Mode
            context = retrieved.to_string(index=False)
            
            if is_recommendation_intent:
                prompt = f"""You are a rigorous and professional CUHK Engineering career advisor. You must ONLY answer based on the real job data provided below.

Student Background:
- Year: {getattr(user_profile, 'year', user_profile.get('year', 'Unknown'))}
- Skills: {getattr(user_profile, 'skills', user_profile.get('skills', []))}
- Interests: {getattr(user_profile, 'interests', user_profile.get('interests', ''))}
- GPA: {getattr(user_profile, 'gpa', user_profile.get('gpa', ''))}

Real Job Data:
{context}

User Question: {user_query}

Please recommend suitable positions and clearly explain why they are suitable for this student."""
            else:
                prompt = f"""You are a professional CUHK Engineering career advisor. Answer based on the real data provided.

Student Background:
- Year: {getattr(user_profile, 'year', user_profile.get('year', 'Unknown'))}
- Skills: {getattr(user_profile, 'skills', user_profile.get('skills', []))}
- Interests: {getattr(user_profile, 'interests', user_profile.get('interests', ''))}
- GPA: {getattr(user_profile, 'gpa', user_profile.get('gpa', ''))}

Real Job Data:
{context}

User Question: {user_query}

Please answer the question clearly based on the provided data."""

        # ==================== Groq Call ====================
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content.strip()

        print("💡 Recommendation & Analysis:")
        print(answer)
        print("\n" + "="*85)

        log_feedback(user_query, user_profile, recommended_ids, "web_auto")

        return answer, retrieved

    except Exception as e:
        print("Error:", str(e))
        return "Sorry, the system is currently busy. Please try again later.", pd.DataFrame()
