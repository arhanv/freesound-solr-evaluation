import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Add parent directory to path so we can import from 'search'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.stats_utils import get_solr_health, get_content_distribution, get_similarity_spaces
from search.generate_and_index_synthetics import cleanup_synthetic
import requests
from search.configs import SOLR_URL

app = FastAPI(title="Freesound Solr Command Center")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health():
    return get_solr_health()

@app.get("/api/stats")
async def stats():
    return get_content_distribution()

@app.get("/api/spaces")
async def spaces():
    return get_similarity_spaces()

@app.post("/api/action/cleanup")
async def action_cleanup():
    try:
        cleanup_synthetic(SOLR_URL)
        return {"status": "success", "message": "Synthetic data cleaned up."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/action/optimize")
async def action_optimize():
    try:
        resp = requests.get(f"{SOLR_URL}/update?optimize=true")
        resp.raise_for_status()
        return {"status": "success", "message": "Index optimization triggered."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Placeholder for Frontend
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("dashboard/frontend/index.html", "r") as f:
        return f.read()
