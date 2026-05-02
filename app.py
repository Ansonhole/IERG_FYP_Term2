from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import traceback
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="CUHK Engineering Recommendation System")

class QueryRequest(BaseModel):
    query: str
    mode: int = 1

# 直接讀取 index.html（避開 Jinja2 問題）
@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        print("無法讀取 index.html:", str(e))
        return HTMLResponse(content="<h1>系統正常運行</h1><p>聊天室載入失敗，請聯絡開發者。</p>")

@app.post("/recommend")
async def recommend(request: QueryRequest):
    try:
        print(f"收到請求 → Mode: {request.mode}, Query: {request.query[:100]}...")

        from src.recommender import generate_recommendation

        if request.mode == 1:
            from src.user_profile import UserProfile
            user_profile = UserProfile(year="3", skills=["coding"], interests="AI", gpa=3.0)
        else:
            user_profile = None

        answer, results = generate_recommendation(request.query, user_profile)
        return JSONResponse({"success": True, "answer": answer})

    except Exception as e:
        print("推薦錯誤:", str(e))
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
