from fastapi import APIRouter, Request, HTTPException
from typing import Optional

router = APIRouter()


@router.get("/query")
def ask(request: Request, query_text: str, k: Optional[int] = None, explain: bool = True):
    """
    Query endpoint for the RAG system.
    Uses the AlzheimerRAGSystem attached to app.state during startup.
    """
    rag_system = getattr(request.app.state, "rag_system", None)
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    result = rag_system.query(user_query=query_text, k=k, explain=explain)
    return {"query": query_text, "result": result}
