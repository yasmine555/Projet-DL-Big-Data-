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
from app.api.endpoints.admin import admin_router
from app.api.endpoints.auth import router as auth_router
from app.api.endpoints.doctor import router as doctor_router
from app.api.endpoints.mri import router as mri_router

from app.core.alzheimer_rag_system import AlzheimerRAGSystem, RAGConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
if not os.getenv("GROQ_API_KEY"):
    logging.getLogger("uvicorn.error").warning("GROQ_API_KEY not set in environment")

# Global RAG system
rag_system_instance = None

# import document processor (best-effort) — log full exception if it fails
docproc = None
try:
    from app.services import document_processor as docproc  # type: ignore
    logger.info("✓ Imported document_processor module")
except Exception as ex:
    logger.error("Failed to import app.services.document_processor — will not auto-build embeddings.")
    # Log full traceback to help debugging import-time failures (missing deps, syntax errors...)
    tb = traceback.format_exc()
    logger.error(tb)
    # Also print to stderr so it's visible in console if logger misconfigured
    print("=== document_processor import traceback ===", file=sys.stderr)
    print(tb, file=sys.stderr)
    docproc = None



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    global rag_system_instance

    
    logger.info(" Starting Alzheimer's RAG System...")

    try:
        config = RAGConfig()
        rag_system_instance = AlzheimerRAGSystem(config)
        app.state.rag_system = rag_system_instance
        logger.info(" RAG system initialized successfully")
    except Exception as e:
        logger.error(f" Failed to initialize RAG system: {e}")
        app.state.rag_system = None

    # Initialize MRI analyzer
    try:
        from app.services.mri_analyzer import initialize_mri_analyzer
        model_path = Path(__file__).parent / "models" / "best_model.h5"
        logger.info(f"Initializing MRI analyzer with model at {model_path}")
        initialize_mri_analyzer(str(model_path))
        logger.info(" MRI analyzer initialized successfully")
    except Exception as e:
        logger.error(f" Failed to initialize MRI analyzer: {e}")
        logger.warning("MRI analysis endpoints will not be available")

    # After RAG init: ensure embeddings / nodes exist; try to build them if missing
    try:
        # determine nodes path from config if present, else fallback to project/data/processed/nodes.pkl
        nodes_path = None
        if hasattr(config, "NODES_PATH"):
            nodes_path = Path(getattr(config, "NODES_PATH"))
        elif hasattr(config, "nodes_path"):
            nodes_path = Path(getattr(config, "nodes_path"))
        else:
            project_root = Path(__file__).resolve().parents[1]
            nodes_path = project_root / "data" / "processed" / "embeddings.npy"

        logger.info("Checking embeddings file at %s", nodes_path)
        if not nodes_path.exists():
            logger.warning("Embeddings file not found at %s — attempting to run document processor", nodes_path)
            if docproc is None:
                logger.error("Document processor module not available; cannot build embeddings automatically.")
            else:
                # Preferred: call module-level main() which exists in document_processor.py
                if hasattr(docproc, "main") and callable(docproc.main):
                    try:
                        logger.info("Calling document_processor.main() to build embeddings")
                        maybe = docproc.main()
                        if asyncio.iscoroutine(maybe):
                            await maybe
                        logger.info("document_processor.main() completed")
                    except Exception as ex:
                        logger.exception("document_processor.main() failed: %s", ex)
                else:
                    logger.warning("document_processor.main() not present; please use DocumentProcessor API or expose a build function.")
        else:
            logger.info("Embeddings file exists — skipping auto-processing.")
    except Exception as e:
        logger.exception("Error while checking/building embeddings: %s", e)

    yield

    # Shutdown
    logger.info("Shutting down RAG system...")


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