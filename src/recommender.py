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
        recommend_keywords = ["推薦", "suggest", "最適合", "suitable", "找工作", "找實習", "recommend", "best for me", "學院推薦"]
        is_recommendation_intent = any(kw in user_query.lower() for kw in recommend_keywords)

        retrieved = retrieve_top_k(user_query, user_profile, top_k=top_k)
        if not retrieved.empty:
            retrieved['id'] = retrieved['id'].fillna('unknown')
            retrieved['id'] = retrieved['id'].astype(str)
            if 'final_score' in retrieved.columns:
                retrieved['final_score'] = retrieved['final_score'].fillna(0.0)
            if 'similarity_score' in retrieved.columns:
                retrieved['similarity_score'] = retrieved['similarity_score'].fillna(0.0)
                
        recommended_ids = retrieved['id'].astype(str).tolist() if not retrieved.empty else []

        print("\n" + "="*85)
        print("🎯 CUHK Engineering Personalized Recommendation System")
        print("="*85 + "\n")

        is_college_mode = isinstance(user_profile, dict) and user_profile.get("mode") == "college"

        # 建立帶分數的詳細 Context
        context_with_score = ""
        for i, (_, row) in enumerate(retrieved.iterrows(), 1):
            score = row.get('final_score', row.get('similarity_score', 0))
            if row.get('type') == "college_choice":
                title = row.get('college', 'Unknown')
                context_with_score += f"{i}. College: {title} | Score: {score:.4f}\n"
                context_with_score += f"   Question: {row.get('question')}\n"
                context_with_score += f"   Answer: {row.get('answer', '')[:300]}...\n\n"
            else:
                title = row.get('job_title', row.get('title', 'Untitled'))
                company = row.get('company_name', row.get('company', 'Unknown'))
                context_with_score += f"{i}. Job: {title} at {company} | Score: {score:.4f}\n"
                context_with_score += f"   Deadline: {row.get('deadline', 'N/A')}\n\n"

        # ==================== Prompt (要求附上 Score) ====================
        if is_college_mode:
            if is_recommendation_intent:
                prompt = f"""You are a professional CUHK college selection advisor.

Student's Preferences:
1. Hostel: {user_profile.get('residential')}
2. Christian Values: {user_profile.get('christian')}
3. GE Interest: {user_profile.get('ge_interest')}
4. Activities: {user_profile.get('activity')}
5. Academic vs Life: {user_profile.get('academic')}
6. Personality: {user_profile.get('personality')}

Here are the retrieved colleges with their matching scores:
{context_with_score}

User Question: {user_query}

Please recommend the most suitable college(s) based on the data above.
For each recommendation, please include its Score and clearly explain why it is suitable.

**Important**: For each recommendation, output in this exact format:

【1】 College Name (Score: X.XXXX)
Explanation: ...

【2】 College Name (Score: X.XXXX)
Explanation: ...

Make each recommendation clear and separate."""
            else:
                recommended_ids = "0"
                prompt = f"""You are a professional CUHK college selection advisor.

Student's Preferences:
1. Hostel: {user_profile.get('residential')}
2. Christian Values: {user_profile.get('christian')}
3. GE Interest: {user_profile.get('ge_interest')}
4. Activities: {user_profile.get('activity')}
5. Academic vs Life: {user_profile.get('academic')}
6. Personality: {user_profile.get('personality')}

Here are the retrieved colleges with their matching scores:
{context_with_score}

User Question: {user_query}

Please answer clearly and explain based on the data."""
        
        else:
            if is_recommendation_intent:
                prompt = f"""You are a professional CUHK Engineering career advisor.

Student Background:
- Year: {getattr(user_profile, 'year', user_profile.get('year', 'Unknown'))}
- Skills: {getattr(user_profile, 'skills', user_profile.get('skills', []))}
- Interests: {getattr(user_profile, 'interests', user_profile.get('interests', ''))}
- GPA: {getattr(user_profile, 'gpa', user_profile.get('gpa', ''))}

Here are the retrieved job opportunities with their matching scores:
{context_with_score}

User Question: {user_query}

Please recommend the most suitable positions. 
For each recommendation, please include its Score and clearly explain why it is a good match for the student.

**Important**: For each recommendation, output in this exact format:

【1】 Job Title at Company (Score: X.XXXX)
Explanation: ...

【2】 Job Title at Company (Score: X.XXXX)
Explanation: ...

Make each recommendation clear and separate."""
            else:
                prompt = f"""You are a professional CUHK Engineering career advisor.

Student Background:
- Year: {getattr(user_profile, 'year', user_profile.get('year', 'Unknown'))}
- Skills: {getattr(user_profile, 'skills', user_profile.get('skills', []))}
- Interests: {getattr(user_profile, 'interests', user_profile.get('interests', ''))}
- GPA: {getattr(user_profile, 'gpa', user_profile.get('gpa', ''))}

Here are the retrieved job opportunities with their matching scores:
{context_with_score}

User Question: {user_query}

Please answer the question clearly based on the provided data."""

        # ==================== Groq Call ====================
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.65,
            max_tokens=1200
        )
        
        answer = response.choices[0].message.content.strip()

        print("💡 Detailed Analysis & Recommendations:")
        print(answer)
        print("\n" + "="*85)

        log_feedback(user_query, user_profile, recommended_ids, "web_auto")

        return answer, retrieved, recommended_ids, is_recommendation_intent

    except Exception as e:
        print("Error:", str(e))
        return "Sorry, the system is currently busy. Please try again later.", pd.DataFrame()
