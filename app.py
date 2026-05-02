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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """如果模板有問題，先返回簡單 HTML 測試"""
    html_content = """
    <h1>CUHK 工程系推薦系統</h1>
    <p>後端已成功運行！</p>
    <p>如果看到這行文字，代表後端正常。</p>
    <hr>
    <p><a href="#" onclick="alert('後端正常運行')">點我測試</a></p>
    """
    return HTMLResponse(content=html_content)

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
        print("錯誤:", str(e))
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
