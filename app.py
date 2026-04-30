from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import traceback

app = FastAPI(title="CUHK Engineering Recommendation System")
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    query: str
    mode: int = 1

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend")
async def recommend(request: QueryRequest):
    try:
        print(f"收到請求 - Mode: {request.mode}, Query: {request.query}")

        from src.recommender import generate_recommendation

        # 簡化處理：先不傳複雜的 user_profile
        if request.mode == 1:
            # Job Mode - 使用簡單的空 profile
            from src.user_profile import UserProfile
            user_profile = UserProfile(year="3", skills=["python"], interests="", gpa=3.0)
        else:
            user_profile = None  # College Mode

        answer, results = generate_recommendation(request.query, user_profile)

        return JSONResponse({
            "success": True,
            "answer": answer,
            "mode": request.mode
        })

    except Exception as e:
        error_detail = traceback.format_exc()
        print("後端錯誤詳情:")
        print(error_detail)
        
        return JSONResponse({
            "success": False,
            "error": str(e),
            "detail": error_detail[-500:]  # 只顯示最後500字，避免太長
        }, status_code=500)

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)