from fastapi import FastAPI
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
    profile_data: list = None   # 用來接收前端收集的答案

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
        print(f"收到請求 → Mode: {request.mode}")

        from src.recommender import generate_recommendation

        # 轉換 profile_data 成後端可用的格式
        if request.mode == 1 and request.profile_data:
            # 工作模式： [year, skills, interests, gpa]
            user_profile = {
                "year": request.profile_data[0],
                "skills": request.profile_data[1].split(",") if len(request.profile_data) > 1 else [],
                "interests": request.profile_data[2] if len(request.profile_data) > 2 else "",
                "gpa": float(request.profile_data[3]) if len(request.profile_data) > 3 else 3.0
            }
        elif request.mode == 2 and request.profile_data:
            # 學院模式：6個答案
            ans = request.profile_data
            user_profile = {
                "mode": "college",
                "residential": ans[0] if len(ans)>0 else "",
                "christian": ans[1] if len(ans)>1 else "",
                "ge_interest": ans[2] if len(ans)>2 else "",
                "activity": ans[3] if len(ans)>3 else "",
                "academic": ans[4] if len(ans)>4 else "",
                "personality": ans[5] if len(ans)>5 else ""
            }
        else:
            user_profile = None

        answer, results, recommended_ids = generate_recommendation(request.query, user_profile)
        
        return JSONResponse({
            "success": True, 
            "answer": answer,
            "recommended_ids": recommended_ids   # 新增這行
        })

    except Exception as e:
        print("後端錯誤:", str(e))
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/evaluate")
async def evaluate(request: dict):
    try:
        user_query = request.get("query", "")
        recommended_ids = request.get("recommended_ids", [])
        ground_truth_ids = request.get("ground_truth_ids", [])

        from src.evaluator import calculate_metrics, save_evaluation_log
        
        metrics = calculate_metrics(recommended_ids, ground_truth_ids, k=5)
        
        save_evaluation_log(user_query, recommended_ids, ground_truth_ids, metrics)
        
        return JSONResponse({
            "success": True,
            "metrics": metrics,
            "message": "Thank you for your evaluation! The quality of this recommendation has been recorded."
        })
        
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
