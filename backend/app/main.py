"""
Fixed Main FastAPI Application
Location: app/main.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import sys
from pathlib import Path
import asyncio
from fastapi import BackgroundTasks, HTTPException
import importlib, traceback, sys
from  groq import  Groq
from dotenv import load_dotenv
from pathlib import Path as _Path
import os 
import sys
sys.path.append(str(_Path(__file__).parent.parent))


# Load .env from likely locations (backend root, app folder, cwd)


from app.api.endpoints.health import router as health_router
from app.api.endpoints.query import query_router
from app.api.endpoints.questionnaire import questionnaire_router
from app.api.endpoints.auth import router as auth_router
from app.api.endpoints.doctor import router as doctor_router
from app.api.endpoints.mri import router as mri_router




logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    
    logger.info("Starting application...")

    # Initialize MRI analyzer
    try:
        from app.services.mri_analyzer import initialize_mri_analyzer
        model_path = Path(__file__).parent / "models" / "best_model_VGG16.h5"
        logger.info(f"Initializing MRI analyzer with model at {model_path}")
        initialize_mri_analyzer(str(model_path))
        logger.info("✓ MRI analyzer initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize MRI analyzer: {e}")
        logger.warning("MRI analysis endpoints will not be available")

    logger.info("Application startup complete")
    
    yield

    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI application
app = FastAPI(
    title="Alzheimer's Intelligent RAG System",
    description="XAI-powered RAG system for Alzheimer's disease assistance",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Setup CORS - CRITICAL for frontend
# Include common dev ports; allow override via FRONT_ORIGINS env
_default_front = "http://localhost:3000,http://localhost:3001,http://localhost:4000,http://localhost:5173,http://localhost:5174"
FRONT_ORIGINS = os.getenv("FRONT_ORIGINS", _default_front).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in FRONT_ORIGINS if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

API_PREFIX = "/api"

# Serve uploads (so savedPath can be a URL)
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Include routers with v1 prefixes (match frontend)
app.include_router(health_router)
app.include_router(auth_router, prefix=API_PREFIX)

app.include_router(doctor_router, prefix=API_PREFIX)
app.include_router(mri_router, prefix=API_PREFIX)
app.include_router(query_router, prefix=API_PREFIX)
app.include_router(questionnaire_router, prefix=API_PREFIX)


# Admin endpoint: trigger reindexing (background)
async def _run_reindex_task(logger, config):
    if docproc is None:
        logger.error("Document processor module missing; cannot reindex.")
        return
    candidate_names = [
        "build_index", "process_documents", "create_embeddings",
        "build_embeddings", "build_nodes", "index_documents", "run"
    ]
    for fn_name in candidate_names:
        fn = getattr(docproc, fn_name, None)
        if callable(fn):
            try:
                logger.info("Triggering document_processor.%s()", fn_name)
                maybe = fn(config) if config is not None else fn()
                if asyncio.iscoroutine(maybe):
                    await maybe
                logger.info("Reindex finished via %s", fn_name)
                return
            except Exception:
                logger.exception("document_processor.%s failed", fn_name)
    logger.error("Reindexing failed: no suitable document_processor function succeeded.")


@app.post("/api/endpoints/admin/reindex", status_code=202)
async def reindex_endpoint(background_tasks: BackgroundTasks):
    """Trigger document processing / embedding build in background."""
    # Try to pass the same RAGConfig instance if available
    cfg = None
    try:
        cfg = RAGConfig()
    except Exception:
        cfg = None
    if docproc is None:
        raise HTTPException(status_code=503, detail="Document processor not available on server")
    background_tasks.add_task(_run_reindex_task, logger, cfg)
    return {"status": "reindex_started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7000, reload=True)