"""Minimal HTTP API for the document classifier (used by the React frontend).

Run from repo root:
    uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
sys.path.insert(0, str(_SRC))

app = FastAPI(title="Document Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@app.get("/")
def root():
    """Open http://127.0.0.1:8000/ in a browser to see API pointers (this is the backend port)."""
    return {
        "service": "Document Classifier API",
        "port": 8000,
        "interactive_docs": "/docs",
        "openapi_json": "/openapi.json",
        "health": "/api/health",
        "classify": {"method": "POST", "path": "/api/classify", "form_field": "file"},
        "web_ui": "Run `npm run dev` in frontend/ — UI is usually http://localhost:5173 (proxies /api to this server).",
    }


@app.get("/api/health")
def health():
    models_dir = _REPO_ROOT / "artifacts" / "models"
    ok = (models_dir / "best_classifier.pkl").is_file() and (models_dir / "lgbm_classifier.pkl").is_file()
    return {"status": "ok" if ok else "degraded", "models_present": ok}


@app.post("/api/classify")
async def classify_upload(file: UploadFile = File(...)):
    name = file.filename or "upload"
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type {suffix or '(none)'}. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    try:
        from classifier import classify
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Classifier failed to load (missing artifacts or dependencies): {e!s}",
        ) from e

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, prefix="dc_", delete=False) as f:
            tmp_path = f.name
            f.write(data)
        result = classify(tmp_path)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if tmp_path and Path(tmp_path).is_file():
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
