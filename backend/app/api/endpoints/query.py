"""
Enhanced API Endpoints with Better Response Format
Location: app/api/endpoints/query.py
"""

from fastapi import APIRouter, Request, HTTPException, Header, Depends
import logging, datetime
from typing import Dict, Any, Optional
from app.models.requests import QueryRequest
from app.models.db import get_db, COL_RESULTS, COL_CONVERSATIONS
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel
from .auth import require_doctor
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))




logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai", tags=["ai"])

# Enhanced response model
class EnhancedResponse(BaseModel):
    """Enhanced response with better structure"""
    answer: str
    confidence: float
    query_type: str
    user_type: str
    
    # Explainability
    reasoning_trace: list
    sources: list
    
    # Metrics
    retrieval_quality: dict
    answer_quality: dict
    trust_metrics: dict
    
    # Guidance
    warnings: list
    recommendations: list
    next_steps: list


@router.post("/query/", response_model=EnhancedResponse)
async def handle_enhanced_query(
    payload: QueryRequest,
    request: Request,
    authorization: str | None = Header(None),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    """
    Enhanced query endpoint with full XAI capabilities
    
    Request:
    {
        "query": "What are the risk factors I can change?",
        "user_type": "patient",
        "context": {
            "age": 65,
            "symptoms_list": ["memory loss"],
            ...
        },
        "explain": true
    }
    
    Response:
    {
        "answer": "Clear, simple explanation...",
        "confidence": 0.85,
        "reasoning_trace": [...],
        "sources": [...],
        "warnings": [...],
        "recommendations": [...]
    }
    """
    
    # Optional doctor context; if token provided and role doctor, use it
    doctor = None
    if authorization:
        try:
            doctor = await require_doctor(authorization=authorization)
        except Exception:
            doctor = None

    # rag = AlzheimerRAGSystem() - REMOVED legacy RAG
    # user_type = UserType(payload.user_type) ... - REMOVED legacy type check
    
    # We proceed directly to AgentOrchestrator which handles logic internally

    # --- AGENTIC INTEGRATION START ---
    
    # We replace the old RAG call with the new AgentOrchestrator
    # Note: We need to import AgentOrchestrator at the top, but for now we do lazy import or assume it's added.
    from app.services.agent_service import AgentOrchestrator
    
    # Context string builder for the agent
    # We might pass the context object directly if the tool supports it, 
    # but the agent tools currently fetch from DB using patient_id.
    # So we just rely on patient_id being passed.
    
    agent = AgentOrchestrator()
    
    try:
        # Run the agent
        # The agent returns the final string answer.
        # Reasoning trace is hidden in LangGraph state, we might want to expose it later.
        agent_response = await agent.run(
            query=payload.query,
            user_role=payload.user_type,
            patient_id=payload.patient_id or "unknown",
            context_type="chatbot"  # Use 'chatbot' mode for research tools
        )
        
        # Map to result structure
        result = {
            "answer": agent_response,
            "confidence": 0.9, # Placeholder until we extract score from agent state
            "query_type": "agentic_processed",
            "sources": ["Agent Tool Execution"], # detailed sources need extraction from state
            "short_summary": agent_response[:100] + "..."
        }

    except Exception as e:
        logger.error(f"Agent Execution Failed: {e}")
        # Fallback to old RAG or error
        result = {
            "answer": "I apologize, but I encountered an error processing your request with the advanced agent.",
            "confidence": 0.0,
            "query_type": "error"
        }

    # --- AGENTIC INTEGRATION END ---

    # Persist conversation and result if patient_id provided
    patient_id = payload.patient_id
    if patient_id:
        doctor_id = doctor["sub"] if doctor else "system"
        
        conv_doc = {
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "query": payload.query,
            "result": result,
            "created_at": datetime.datetime.utcnow()
        }
        await db[COL_CONVERSATIONS].insert_one(conv_doc)
        
        res_doc = {
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "summary": result.get("short_summary"),
            "answer": result.get("answer"),
            "confidence": result.get("confidence"),
            "query_type": result.get("query_type"),
            "sources": result.get("sources", []),
            "created_at": datetime.datetime.utcnow()
        }
        await db[COL_RESULTS].insert_one(res_doc)

    return {
        "answer": result.get("answer", ""),
        "confidence": result.get("confidence", 0.0),
        "query_type": result.get("query_type", "general"),
        "user_type": result.get("user_type", "patient"),
        "reasoning_trace": result.get("reasoning_trace", []),
        "sources": result.get("sources", []),
        "retrieval_quality": result.get("retrieval_metrics", {}),
        "answer_quality": {"length": len(result.get("answer", ""))},
        "trust_metrics": {"patient_context_used": result.get("patient_context_used", False)},
        "warnings": result.get("warnings", []),
        "recommendations": result.get("recommendations", []),
        "next_steps": []
    }


@router.post("/evaluate")
async def evaluate_response(request: Request, payload: Dict[str, Any]):
   
    
    from app.services.response_evaluator import Evaluator
    
    query = payload.get("query")
    response = payload.get("response")
    retrieved_docs = payload.get("retrieved_docs", [])
    stated_confidence = payload.get("stated_confidence", 0.5)
    user_type = payload.get("user_type", "patient")
    
    if not query or not response:
        raise HTTPException(status_code=400, detail="query and response required")
    
    # Run comprehensive evaluation
    evaluator = Evaluator()
    
    try:
        metrics = evaluator.evaluate_comprehensive(
            query=query,
            response=response,
            retrieved_docs=retrieved_docs,
            stated_confidence=stated_confidence,
            user_type=user_type,
            k=10
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
    
    # Format response
    return {
        "overall_quality": metrics.overall_quality_score,
        "trust_level": metrics.trust_level,
        "detailed_metrics": {
            "retrieval": {
                "precision": metrics.retrieval_precision,
                "recall": metrics.retrieval_recall,
                "f1": metrics.retrieval_f1,
                "mrr": metrics.mrr,
                "ndcg": metrics.ndcg
            },
            "answer": {
                "relevance": metrics.answer_relevance,
                "completeness": metrics.answer_completeness,
                "coherence": metrics.answer_coherence,
                "readability": metrics.answer_readability
            },
            "explainability": {
                "citation_accuracy": metrics.citation_accuracy,
                "source_utilization": metrics.source_utilization,
                "confidence_calibration": metrics.confidence_calibration
            },
            "patient_friendliness": {
                "language_simplicity": metrics.language_simplicity,
                "medical_jargon_ratio": metrics.medical_jargon_ratio,
                "actionability": metrics.actionability
            }
        },
        "strengths": metrics.strengths,
        "weaknesses": metrics.weaknesses,
        "improvement_suggestions": metrics.improvement_suggestions
    }


# Expose router
query_router = router