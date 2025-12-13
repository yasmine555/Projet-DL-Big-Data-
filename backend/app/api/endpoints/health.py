from fastapi import APIRouter, Request
from app.models.responses import HealthCheckResponse

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("/", response_model=HealthCheckResponse)
async def health_check(request: Request):
    """Check system health status (does not raise if RAG missing)"""
    rag = getattr(request.app.state, "rag_system", None)
    return HealthCheckResponse(
        status="healthy",
        rag_system="initialized" if rag else "not_initialized",
        models_loaded=rag is not None
    )

@router.get("/ready")
async def readiness_check():
    """Check if service is ready"""
    return {"status": "ready"}

@router.get("/live")
async def liveness_check():
    """Check if service is alive"""
    return {"status": "alive"}

health_router = router
