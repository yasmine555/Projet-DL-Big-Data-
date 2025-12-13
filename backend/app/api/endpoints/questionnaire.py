from fastapi import APIRouter, Request, HTTPException, Header, Depends
from typing import List, Optional
from app.models.requests import QuestionnaireRequest
from app.models.responses import QuestionnaireResponse, SourceInfo, EvaluationMetrics
from app.core.alzheimer_rag_system import PatientContext, UserType, AlzheimerRAGSystem
from app.services.ragas_evaluator import RAGASEvaluator
from app.models.db import get_db, COL_RESULTS
from motor.motor_asyncio import AsyncIOMotorDatabase
from .auth import require_doctor
import datetime
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# GET /results/{patient_id}
@router.get("/results/{patient_id}")
async def get_patient_results(
    patient_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    authorization: str | None = Header(None)
):
    """
    Retrieve the latest questionnaire/analysis result for a specific patient.
    """
    try:
        # Find the most recent result for this patient
        # Sort by created_at descending
        result = await db[COL_RESULTS].find_one(
            {"patient_id": patient_id},
            sort=[("created_at", -1)]
        )
        
        if not result:
            # Return empty list to match frontend expectation (api.js expects array)
            return []
            
        # Transform to match frontend expectation
        # api.js expects: { answer, confidence, key_findings, recommendations, citations: [{title,url,snippet}] }
        # But the frontend actually handles the raw result object if it matches the structure.
        # Let's return a list containing the result document.
        
        # Convert ObjectId to string if needed (though find_one usually returns dict)
        if "_id" in result:
            result["_id"] = str(result["_id"])
            
        return [result]
        
    except Exception as e:
        logger.error(f"Error retrieving results for patient {patient_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# POST /submit
@router.post("/submit", response_model=QuestionnaireResponse)
async def submit_questionnaire(
    payload: QuestionnaireRequest,
    request: Request,
    authorization: str | None = Header(None),
    db: AsyncIOMotorDatabase = Depends(get_db)
):
    """
    Accepts: { responses: [...], metadata: {...} }
    Returns: structured risk assessment and a RAG-generated summary (if RAG available)
    """
    # Basic validation (Pydantic ensures non-empty responses)
    if not payload.responses or len(payload.responses) == 0:
        raise HTTPException(status_code=400, detail="Responses are required")

    # Build patient context
    md = payload.metadata
    patient_ctx = PatientContext(
        age=md.age,
        symptoms=md.symptoms_list or [],
        medical_history=md.medical_history or [],
        current_medications=md.current_medications or [],
        mmse_score=md.mmse_score,
        moca_score=md.moca_score,
        biomarkers=md.biomarkers.dict() if hasattr(md.biomarkers, "dict") else (md.biomarkers or {})
    )

    # Simple risk heuristic: proportion positive answers (assumes binary 0/1)
    try:
        raw_score = (sum(int(bool(x)) for x in payload.responses) / len(payload.responses)) * 100.0
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid response values")

    normalized = max(0.0, min(1.0, raw_score / 100.0))
    interpretation = "low" if normalized < 0.33 else ("medium" if normalized < 0.66 else "high")

    # Attempt to call RAG system for a short clinical summary and evidence
    rag = getattr(request.app.state, "rag_system", None)
    summary_text = None
    sources_list: List[SourceInfo] = []
    eval_metrics = None
    rag_result = {}

    if rag is not None:
        # Updated query for natural response
        query_text = (
            "Generate a natural, coherent clinical summary for this patient profile, including "
            "insights on possible diagnoses, risks, and personalized recommendations. Respond in "
            "flowing paragraphs without forced headings."
        )
        rag_result = rag.query(
            user_query=query_text,
            user_type=UserType.PATIENT if (md.user_type or "patient") == "patient" else UserType.DOCTOR,
            patient_context=patient_ctx,
            explain=True
        )
        summary_text = rag_result.get("answer") or rag_result.get("summary")
        # Build sources
        for s in rag_result.get("evidence", []):
            sources_list.append(SourceInfo(
                title=s.get("document_title", "Unknown"),
                page=s.get("page_number", None),
                relevance=s.get("score", 0.0),
                type=s.get("content_type", "text")
            ))
        # Evaluate the RAG response using SimpleEvaluator (if available)
        try:
            # evaluator = Evaluator() # Evaluator not imported/defined in this scope
            # eval_res = evaluator.evaluate(query_text, summary_text or "", rag_result.get("evidence", []), k=5)
            # eval_metrics = eval_res.__dict__
            pass
        except Exception:
            eval_metrics = None

    # Build response payload
    response = QuestionnaireResponse(
        risk_score=raw_score,
        normalized=normalized,
        interpretation=interpretation,
        summary=summary_text or f"Preliminary risk appears {interpretation}. Consider clinical assessment for confirmation.",
        details={"patient_context": patient_ctx.__dict__, "evaluator": eval_metrics} if eval_metrics else {"patient_context": patient_ctx.__dict__},
        sources=sources_list
    )
    
    # Persist questionnaire result
    # We try to persist if we have a patient_id, regardless of doctor auth (for now) to ensure data is saved.
    # But ideally we should link it to a doctor if possible.
    try:
        doctor_id = "system"
        if authorization:
            try:
                # Try to get user/doctor info but don't block if it fails
                # We can use a simpler token decode or just assume system if auth fails
                # For now, let's try to get the doctor if we can
                # doctor = await require_doctor(authorization=authorization)
                # doctor_id = doctor["sub"]
                pass
            except Exception:
                pass
        
        # Check for patient_id in payload or metadata
        patient_id = payload.patient_id or (md.extra.get("patient_id") if md and md.extra else None)
        
        if patient_id:
            logger.info(f"Persisting result for patient {patient_id}")
            res_doc = {
                "patient_id": patient_id,
                "doctor_id": doctor_id,
                "summary": (summary_text or response.summary),
                "answer": summary_text or None,
                "confidence": response.normalized,
                "query_type": "questionnaire",
                "risk_score": response.risk_score,
                "interpretation": response.interpretation,
                "sources": [s.dict() for s in sources_list],
                "reasoning_trace": rag_result.get("reasoning_trace", []),
                "created_at": datetime.datetime.utcnow()
            }
            await db[COL_RESULTS].insert_one(res_doc)
            logger.info("Result persisted successfully")
        else:
            logger.warning("No patient_id provided, result not persisted")
            
    except Exception as e:
        logger.error(f"Failed to persist questionnaire result: {e}")
        # Do not fail the API on persistence errors
        pass

    return response

# expose router
questionnaire_router = router