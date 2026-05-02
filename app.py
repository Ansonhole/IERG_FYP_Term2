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
    """使用美觀聊天室模板"""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        # 如果模板出問題，fallback 到簡單頁面
        print("模板載入失敗，使用簡單頁面:", str(e))
        html = "<h1>系統正常運行</h1><p>聊天室模板載入失敗，請檢查 templates/index.html</p>"
        return HTMLResponse(content=html)

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
