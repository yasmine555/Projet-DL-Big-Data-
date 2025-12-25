from fastapi import APIRouter, Request, HTTPException, Header, Depends
from typing import List, Optional
from app.models.requests import QuestionnaireRequest
from app.models.responses import QuestionnaireResponse, SourceInfo, EvaluationMetrics


from app.models.db import get_db, COL_RESULTS
from motor.motor_asyncio import AsyncIOMotorDatabase
from .auth import require_doctor
import datetime
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
from dotenv import load_dotenv
load_dotenv()


@router.get("/doctor/patient/result/{patient_id}")
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
            return []
            
        # Helper to serialize Mongo types
        def serialize_doc(doc):
            if not doc: return doc
            doc["_id"] = str(doc["_id"])
            for k, v in doc.items():
                if isinstance(v, (datetime.datetime, datetime.date)):
                    doc[k] = v.isoformat()
                if isinstance(v, ObjectId):
                    doc[k] = str(v)
            return doc

        # Serialize results
        serialized_result = serialize_doc(result)
        return [serialized_result]
        
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
    Accepts: { responses: [...], metadata: {...}, patient_id: ... }
    Returns: structured risk assessment and summary
    """
    # Basic validation
    if not payload.responses or len(payload.responses) == 0:
        raise HTTPException(status_code=400, detail="Responses are required")

    # Build patient context dict
    md = payload.metadata
    patient_context = {
        "age": md.age,
        "sex": md.sex,
        "symptoms_list": md.symptoms_list or [],
        "medical_history": md.medical_history or [],
        "current_medications": md.current_medications or [],
        "mmse_score": md.mmse_score,
        "moca_score": md.moca_score,
        "biomarkers": md.biomarkers.dict() if hasattr(md.biomarkers, "dict") else (md.biomarkers or {}),
        "imaging_findings": md.imaging_findings,
        "education_level": md.education_level,
        "family_history": md.family_history
    }

    # Simple risk heuristic: proportion positive answers
    try:
        raw_score = (sum(int(bool(x)) for x in payload.responses) / len(payload.responses)) * 100.0
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid response values")

    normalized = max(0.0, min(1.0, raw_score / 100.0))
    interpretation = "low" if normalized < 0.33 else ("medium" if normalized < 0.66 else "high")

    # Generate AI summary using direct API call (bypass agent for performance)
    from openai import OpenAI
    import os
    
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )
    
    # Build concise patient summary prompt
    summary_prompt = f"""You are a clinical decision support assistant.

Generate a CONCISE clinical summary (4-5 sentences maximum) for this patient:

Patient Information:
- Age: {md.age or 'Not provided'}
- Sex: {md.sex or 'Not provided'}
- Symptoms: {', '.join(md.symptoms_list) if md.symptoms_list else 'None reported'}
- MMSE Score: {md.mmse_score if md.mmse_score is not None else 'Not tested'}
- MoCA Score: {md.moca_score if md.moca_score is not None else 'Not tested'}
- MRI Finding: {md.imaging_findings or 'No imaging yet'}
- Medical History: {', '.join(md.medical_history) if md.medical_history else 'None'}
- Risk Level: {interpretation} ({raw_score:.1f}%)

Provide:
1. Key clinical findings
2. Most likely concern or diagnosis
3. Top 2-3 recommendations

Be direct and actionable."""

    # Extract patient_id early for logging
    patient_id = payload.patient_id or (md.extra.get("patient_id") if md and md.extra else "unknown")

    logger.info(f"Generating AI summary for patient {patient_id}")
    try:
        response_ai = client.chat.completions.create(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            temperature=0.2,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": "You are a compassionate medical assistant providing concise clinical insights."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        summary_text = response_ai.choices[0].message.content
        logger.info("AI summary generated successfully")
    except Exception as e:
        logger.error(f"AI summary generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI summary generation failed: {str(e)}")

    # Determine if we need to create a new patient (anonymous/temp user)
    # Check if patient_id is missing, "unknown", or starts with "temp_"
    created_patient_id = None
    if not patient_id or patient_id == "unknown" or str(patient_id).startswith("temp_"):
        logger.info(f"Creating new patient record for anonymous user (id provided: {patient_id})")
        new_patient_doc = {
            "name": md.name or "Anonymous Patient",
            "age": md.age,
            "sex": md.sex,
            "medical_history": ', '.join(md.medical_history) if md.medical_history else "",
            "family_history": str(md.family_history) if md.family_history else "",
            "symptoms_list": md.symptoms_list,
            "education_level": md.education_level,
            "mmse_score": md.mmse_score,
            "moca_score": md.moca_score,
            "biomarkers": md.biomarkers.dict() if hasattr(md.biomarkers, "dict") else {},
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow(),
            "doctor_id": "system",
            "is_anonymous": True
        }
        
        try:
            from app.models.db import COL_PATIENTS
            new_p = await db[COL_PATIENTS].insert_one(new_patient_doc)
            patient_id = str(new_p.inserted_id)
            created_patient_id = patient_id
            logger.info(f"Created new patient with ID: {patient_id}")
        except Exception as e:
            logger.error(f"Failed to create patient record: {e}")
            # Fallback to temp ID if insertion fails, but this defeats the purpose
            pass

    # Build response
    response = QuestionnaireResponse(
        risk_score=raw_score,
        normalized=normalized,
        interpretation=interpretation,
        summary=summary_text,
        details={"patient_context": patient_context},
        sources=[], # No RAG sources for now
        created_patient_id=created_patient_id
    )
    
    # Persist result
    # Persist result
    logger.info("Attempting to persist result...")
    try:
        doctor_id = "system"
        if authorization:
            try:
                doctor = await require_doctor(authorization=authorization)
                doctor_id = doctor["sub"]
            except Exception as e:
                logger.warning(f"Failed to get doctor info from token: {e}")
        
        # patient_id was already extracted earlier
        
        if patient_id and patient_id != "unknown":
            logger.info(f"Persisting result for patient {patient_id}")
            res_doc = {
                "patient_id": patient_id,
                "doctor_id": doctor_id,
                "summary": summary_text,
                "answer": summary_text,
                "confidence": normalized,
                "query_type": "questionnaire",
                "risk_score": raw_score,
                "interpretation": interpretation,
                "sources": [],
                "reasoning_trace": [],
                "created_at": datetime.datetime.utcnow()
            }
            inserted = await db[COL_RESULTS].insert_one(res_doc)
            logger.info(f"Result persisted successfully with ID {inserted.inserted_id}")
        else:
            logger.warning(f"No valid patient_id provided (got '{patient_id}'), result not persisted")
            
    except Exception as e:
        logger.error(f"Failed to persist questionnaire result: {e}")
        # Do not fail the API on persistence errors. Log it loudly.

    return response

    return response

# expose router
questionnaire_router = router