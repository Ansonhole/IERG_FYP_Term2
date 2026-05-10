from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import traceback
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CUHK Engineering Recommendation System")

class QueryRequest(BaseModel):
    query: str
    mode: int = 1
    college_profile: list = None   # 用來接收學院模式的6個答案

@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except:
        return HTMLResponse(content="<h1>系統正常運行</h1>")

@app.post("/recommend")
async def recommend(request: QueryRequest):
    try:
        print(f"收到請求 → Mode: {request.mode}, Query: {request.query[:80]}...")

        from src.recommender import generate_recommendation

        if request.mode == 1:  # 工作推薦模式
            from src.user_profile import UserProfile
            user_profile = UserProfile(
                year="3", 
                skills=["coding"], 
                interests="AI, software engineering", 
                gpa=3.0
            )
        else:  # 學院推薦模式
            # 將前端傳來的 list 轉成 dict
            answers = request.college_profile or ["無所謂"] * 6
            user_profile = {
                "mode": "college",
                "residential": answers[0] if len(answers) > 0 else "無所謂",
                "christian": answers[1] if len(answers) > 1 else "無所謂",
                "ge_interest": answers[2] if len(answers) > 2 else "中",
                "activity": answers[3] if len(answers) > 3 else "無所謂",
                "academic": answers[4] if len(answers) > 4 else "無所謂",
                "personality": answers[5] if len(answers) > 5 else "無所謂"
            }

        answer, results = generate_recommendation(request.query, user_profile)
        
        return JSONResponse({"success": True, "answer": answer})

    except Exception as e:
        print("後端錯誤:", str(e))
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
