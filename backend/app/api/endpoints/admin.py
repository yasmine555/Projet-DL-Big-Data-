from fastapi import APIRouter, Request, HTTPException, Query
from typing import List, Dict, Any

import traceback
import logging

from typing import Union

router = APIRouter(prefix="/admin", tags=["Admin"])
logger = logging.getLogger(__name__)

@router.get("/chunks/sample")
async def sample_chunks(request: Request, limit: int = Query(20, ge=1, le=100)):
    """Return a sample of processed chunks (text + metadata)."""
    rag = getattr(request.app.state, "rag_system", None)
    nodes = []
    if rag is not None and hasattr(rag, "retriever") and getattr(rag.retriever, "nodes", None):
        nodes = rag.retriever.nodes
    else:
        # fallback: try to load nodes.pkl from configured path
        cfg = RAGConfig()
        p = cfg.embeddings_path
        import pickle, os
        if os.path.exists(p):
            with open(p, "rb") as f:
                nodes = pickle.load(f)
    if not nodes:
        raise HTTPException(status_code=404, detail="No chunks/nodes available")
    out = []
    for i, n in enumerate(nodes[:limit]):
        md = getattr(n, "metadata", {}) or {}
        out.append({
            "index": i,
            "text_snippet": (n.text[:800] + "...") if len(n.text) > 800 else n.text,
            "metadata": md
        })
    return {"count": len(out), "samples": out}

@router.post("/evaluate")
async def evaluate_query(request: Request, payload: Dict[str, Any]):
    """
    Run RAG for a query and evaluate the returned answer using SimpleEvaluator.
    Payload: { "query": "...", "user_type": "patient" }
    """
    query_text = payload.get("query")
    if not query_text:
        raise HTTPException(status_code=400, detail="query required")

    rag = getattr(request.app.state, "rag_system", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG system not available")

    try:
        # Normalize user_type (accept string or enum) and call RAG safely
        raw_ut: Union[str, UserType, None] = payload.get("user_type", "patient")
        if isinstance(raw_ut, UserType):
            user_type_enum = raw_ut
        else:
            # try common conversions, default to PATIENT
            try:
                user_type_enum = UserType(str(raw_ut).lower())
            except Exception:
                try:
                    user_type_enum = UserType[str(raw_ut).upper()]
                except Exception:
                    user_type_enum = UserType.PATIENT

        rag_result = rag.query(user_query=query_text, user_type=user_type_enum, patient_context=None, explain=True)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("RAG query failed: %s\n%s", e, tb)
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")

    # Normalize answer/evidence
    answer = ""
    if isinstance(rag_result, dict):
        answer = rag_result.get("answer") or rag_result.get("summary") or ""
        evidence = rag_result.get("evidence", []) or []
    else:
        # Unexpected shape
        answer = str(rag_result)
        evidence = []

    # Ensure evidence items have numeric 'score'
    normalized_docs = []
    for d in evidence:
        try:
            score = float(d.get("score", d.get("relevance_score", 0.0)))
        except Exception:
            score = 0.0
        normalized_docs.append({"score": score, **{k: v for k, v in d.items() if k != "score" and k != "relevance_score"}})

    # If no evidence, evaluator can still run but will compute low metrics
    try:
        evaluator = Evaluator()
        metrics = evaluator.evaluate(query_text, answer, normalized_docs, k=min(10, max(1, len(normalized_docs))))
        # Convert dataclass to dict
        metrics_dict = metrics.__dict__ if hasattr(metrics, "__dict__") else dict(metrics)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Evaluator failed: %s\n%s", e, tb)
        metrics_dict = {"error": str(e), "trace": tb}

    return {"rag_result": rag_result, "evaluation": metrics_dict}

admin_router = router