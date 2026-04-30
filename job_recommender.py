from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json

load_dotenv()

# 載入三個獨立資料檔案
def load_all_data():
    data = []
    
    # 載入實習資料
    try:
        with open("internships.json", "r", encoding="utf-8") as f:
            internships = json.load(f)
            for item in internships:
                item["type"] = "internship"
                data.append(item)
        print(f"✅ 已載入 {len(internships)} 筆實習資料")
    except FileNotFoundError:
        print("⚠️ 找不到 internships.json 檔案")
    
    # 載入畢業職位資料
    try:
        with open("graduate_jobs.json", "r", encoding="utf-8") as f:
            jobs = json.load(f)
            for item in jobs:
                item["type"] = "graduate_job"
                data.append(item)
        print(f"✅ 已載入 {len(jobs)} 筆畢業職位資料")
    except FileNotFoundError:
        print("⚠️ 找不到 graduate_jobs.json 檔案")
    
    # 載入學院選擇資料
    try:
        with open("college_choices.json", "r", encoding="utf-8") as f:
            colleges = json.load(f)
            for item in colleges:
                item["type"] = "college_choice"
                data.append(item)
        print(f"✅ 已載入 {len(colleges)} 筆學院選擇資料")
    except FileNotFoundError:
        print("⚠️ 找不到 college_choices.json 檔案")
    
    return data

# 建立知識庫文字
def build_knowledge_base(data):
    knowledge = """你是一位專為 CUHK 工程系學生服務的職業與升學顧問。
請嚴格根據以下提供的資料回答所有問題，不要編造任何不存在的資訊。
如果問題不在資料範圍內，請禮貌地告訴用戶並引導回實習、畢業職位或學院選擇的話題。

=== CUHK 工程系推薦資料庫 ===

"""
    for item in data:
        if item.get("type") == "internship":
            knowledge += f"[實習]\n職位：{item.get('title')}\n公司：{item.get('company')}\n地點：{item.get('location')}\n要求：{item.get('requirement')}\n福利：{item.get('benefit')}\n申請連結：{item.get('apply_link', '未提供')}\n"
            knowledge += "-" * 70 + "\n\n"
        
        elif item.get("type") == "graduate_job":
            knowledge += f"[畢業職位]\n職位：{item.get('title')}\n公司：{item.get('company')}\n地點：{item.get('location')}\n要求：{item.get('requirement')}\n福利：{item.get('benefit')}\n申請連結：{item.get('apply_link', '未提供')}\n"
            knowledge += "-" * 70 + "\n\n"
        
        elif item.get("type") == "college_choice":
            knowledge += f"[學院/大學選擇]\n名稱：{item.get('college')}\n描述：{item.get('description')}\n優勢：{item.get('advantage')}\n適合對象：{item.get('suit_for')}\n"
            knowledge += "-" * 70 + "\n\n"
    
    return knowledge

# ==================== 主程式 ====================
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

all_data = load_all_data()
system_prompt = build_knowledge_base(all_data)

print("\n🎉 系統準備完成！模型已學習你的三種類型資料。")
print("你可以開始自然聊天，例如：")
print("   • 有哪些適合 CUHK 工程系的 AI 實習？")
print("   • HSBC 畢業職位的起薪大約多少？")
print("   • CUHK Computer Science 學院適合對 AI 有興趣的學生嗎？")
print("輸入 'exit' 或 'quit' 結束\n")

history = []

while True:
    user_input = input("你： ")
    if user_input.lower() in ["exit", "quit"]:
        print("再見！祝你的 FYP 順利完成～")
        break

    # 組合 prompt
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=system_prompt)])] + history
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )

    answer = response.text.strip()
    print(f"推薦顧問：{answer}\n")

    # 更新對話歷史
    history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))
    history.append(types.Content(role="model", parts=[types.Part.from_text(text=answer)]))

    # 限制歷史長度，避免超過限制
    if len(history) > 30:
        history = history[-30:]