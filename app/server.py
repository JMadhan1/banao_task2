"""Kestrel claim-review service.

    uvicorn app.server:app --port 8000        (from task2/)

POST /score   one claim as JSON -> fraud probability, action, reasons
GET  /        the review screen
GET  /health  liveness + which model is loaded

No paid API or key is needed: the model runs locally.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from src.scoring import Scorer

app = FastAPI(title="Kestrel warranty claim review")
STATIC = Path(__file__).parent / "static"

try:
    scorer = Scorer()
    load_error = None
except FileNotFoundError:
    scorer = None
    load_error = "Model files not found. Run `python -m src.train` once, then restart."


@app.get("/health")
def health():
    if scorer is None:
        return JSONResponse({"status": "error", "detail": load_error}, status_code=503)
    return {"status": "ok", "model_trained_on_claims_until": scorer.trained_until}


@app.post("/score")
def score(record: dict):
    if scorer is None:
        return JSONResponse({"error": load_error}, status_code=503)
    errors = scorer.validate(record)
    if errors:
        return JSONResponse({"error": "invalid claim", "details": errors}, status_code=422)
    try:
        return scorer.score(record)
    except Exception as e:  # never show a stack trace to the desk
        return JSONResponse({"error": f"could not score this claim: {e}"}, status_code=500)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
