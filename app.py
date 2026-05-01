from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import traceback
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CUHK Engineering Recommendation System")
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    query: str
    mode: int = 1

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """首頁 - 渲染聊天室介面"""
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}   # 正確寫法：只傳一個 dict
    )

@app.post("/recommend")
async def recommend(request: QueryRequest):
    """接收前端請求並呼叫推薦邏輯"""
    try:
        print(f"收到請求 → Mode: {request.mode}, Query: '{request.query}'")

        from src.recommender import generate_recommendation

        # 簡單建立 user_profile（工作模式使用）
        if request.mode == 1:
            from src.user_profile import UserProfile
            user_profile = UserProfile(year="3", skills=["coding"], interests="AI", gpa=3.0)
        else:
            user_profile = None

        answer, results = generate_recommendation(request.query, user_profile)

        return JSONResponse({
            "success": True,
            "answer": answer
        })

    except Exception as e:
        error_detail = traceback.format_exc()
        print("後端錯誤詳情:\n", error_detail)
        
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
