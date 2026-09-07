from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline import analyze_well

app = FastAPI(title="Advanced Oilfield Production Surveillance Engine")

class AnalyzeRequest(BaseModel):
    well_id: str
    horizon_months: int = 3
    as_of_date: str | None = None

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        return analyze_well(req.well_id, req.horizon_months, req.as_of_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}