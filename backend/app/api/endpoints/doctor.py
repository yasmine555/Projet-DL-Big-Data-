from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.models.requests import PatientCreate, PatientUpdate
from app.models.responses import PatientOut
from app.models import db
from app.api.endpoints.auth import require_doctor
from datetime import datetime

router = APIRouter(prefix="/doctor", tags=["doctor"])

def _build_patient_out(d: dict) -> PatientOut:
    """Helper function to build PatientOut from database document"""
    return PatientOut(
        id=str(d.get("_id")),
        name=d.get("name"),
        age=d.get("age"),
        sex=d.get("sex"),
        education_level=d.get("education_level"),
        years_of_education=d.get("years_of_education"),
        medical_history=d.get("medical_history"),
        family_history=d.get("family_history"),
        mse=d.get("mse"),
        moca=d.get("moca"),
        mmse_score=d.get("mmse_score") or d.get("mse"),  # Try mmse_score first, fall back to mse
        moca_score=d.get("moca_score") or d.get("moca"),  # Try moca_score first, fall back to moca
        symptoms_list=d.get("symptoms_list"),
        biomarkers=d.get("biomarkers"),
        imaging_findings=d.get("imaging_findings"),
        neuro_exam_notes=d.get("neuro_exam_notes"),
        metrics=d.get("metrics"),
        created_at=d.get("created_at"),
    )


@router.get("/patients", response_model=List[PatientOut])
async def list_patients(doctor = Depends(require_doctor)):
    """Get all patients for the logged-in doctor only"""
    doctor_id = doctor["sub"]
    
    # Filter patients by doctor_id
    docs = await db.patients_find({"doctor_id": doctor_id})
    out = [_build_patient_out(d) for d in docs]
    return out

@router.post("/patients", response_model=PatientOut)
async def create_patient(body: PatientCreate, doctor = Depends(require_doctor)):
    """Create a new patient and associate with the logged-in doctor"""
    doctor_id = doctor["sub"]
    
    data = body.dict(exclude_unset=True)
    # Add doctor_id and timestamps
    data["doctor_id"] = doctor_id
    data["created_at"] = datetime.utcnow()
    data["updated_at"] = datetime.utcnow()
    
    created = await db.patients_insert_one(data)
    d = await db.patients_find_one({"_id": created.inserted_id})
    return _build_patient_out(d)

@router.get("/patients/{patient_id}", response_model=PatientOut)
async def get_patient(patient_id: str, doctor = Depends(require_doctor)):
    """Get a specific patient - only if owned by the logged-in doctor"""
    doctor_id = doctor["sub"]
    
    # Verify patient belongs to this doctor
    d = await db.patients_find_one({"_id": db.to_object_id(patient_id), "doctor_id": doctor_id})
    if not d:
        raise HTTPException(status_code=404, detail="Patient not found or access denied")
    
    return _build_patient_out(d)


@router.put("/patients/{patient_id}", response_model=PatientOut)
async def update_patient(patient_id: str, body: PatientUpdate, doctor = Depends(require_doctor)):
    """Update a patient - only if owned by the logged-in doctor"""
    doctor_id = doctor["sub"]
    
    # Verify patient belongs to this doctor
    existing = await db.patients_find_one({"_id": db.to_object_id(patient_id), "doctor_id": doctor_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Patient not found or access denied")
    
    data = body.dict(exclude_unset=True)
    data["updated_at"] = datetime.utcnow()
    
    ok = await db.patients_update_one({"_id": db.to_object_id(patient_id)}, {"$set": data})
    if not ok:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    d = await db.patients_find_one({"_id": db.to_object_id(patient_id)})
    return _build_patient_out(d)

@router.delete("/patients/{patient_id}")
async def delete_patient(patient_id: str, doctor = Depends(require_doctor)):
    """Delete a patient - only if owned by the logged-in doctor"""
    doctor_id = doctor["sub"]
    
    # Verify patient belongs to this doctor before deleting
    existing = await db.patients_find_one({"_id": db.to_object_id(patient_id), "doctor_id": doctor_id})
    if not existing:
        raise HTTPException(status_code=404, detail="Patient not found or access denied")
    
    result = await db.patients_delete_one({"_id": db.to_object_id(patient_id), "doctor_id": doctor_id})
    if not result.deleted_count:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"ok": True}

@router.get("/patients/{patient_id}/conversations")
async def get_patient_conversations(patient_id: str, doctor = Depends(require_doctor)):
    """Get all conversations for a specific patient - only if owned by doctor"""
    from app.models.db import get_db, COL_CONVERSATIONS
    doctor_id = doctor["sub"]
    database = await get_db()
    
    # Verify patient belongs to this doctor
    patient = await db.patients_find_one({"_id": db.to_object_id(patient_id), "doctor_id": doctor_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found or access denied")
    
    # Find all conversations for this patient
    cursor = database[COL_CONVERSATIONS].find(
        {"patient_id": patient_id}
    ).sort("created_at", -1)
    
    conversations = []
    async for conv in cursor:
        # Convert ObjectId to string for JSON serialization
        conv["_id"] = str(conv["_id"])
        # Convert datetime to ISO format
        if conv.get("created_at"):
            conv["created_at"] = conv["created_at"].isoformat()
        conversations.append(conv)
    
    return {"conversations": conversations}

