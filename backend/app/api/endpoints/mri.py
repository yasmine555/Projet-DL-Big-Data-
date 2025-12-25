from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List
from datetime import datetime
import os
import uuid
from pathlib import Path
import logging

from app.api.endpoints.auth import require_doctor
from app.models import db
from app.services.mri_analyzer import get_mri_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mri", tags=["mri"])

# Upload directory for MRI scans
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads" / "mri"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/analyze-patient/{patient_id}")
async def analyze_patient_mri(
    patient_id: str,
    file: UploadFile = File(...),
    doctor=Depends(require_doctor)
):
    """
    Upload and analyze an MRI scan for a patient.
    Saves the file, runs ML analysis, stores metadata in DB, and updates patient record.
    """
    doctor_id = doctor["sub"]
    
    # Verify patient belongs to this doctor
    patient = await db.patients_find_one({
        "_id": db.to_object_id(patient_id),
        "doctor_id": doctor_id
    })
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found or access denied")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1] if file.filename else ".png"
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = UPLOAD_DIR / unique_filename
        
        # Save file to disk
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Generate URL for the saved file
        image_url = f"/uploads/mri/{unique_filename}"
        
        # Analyze the MRI using the ML model
        try:
            analyzer = get_mri_analyzer()
            analysis_result = analyzer.predict(file_content)
            
            # Save XAI image if available
            xai_url = None
            if analysis_result.get("xai_image") is not None:
                from PIL import Image as PILImage
                xai_img_np = analysis_result["xai_image"]
                xai_pil = PILImage.fromarray(xai_img_np)
                
                xai_filename = f"xai_{unique_filename.split('.')[0]}.png"
                xai_path = UPLOAD_DIR / xai_filename
                xai_pil.save(xai_path)
                xai_url = f"/uploads/mri/{xai_filename}"
                
        except Exception as e:
            logger.error(f"MRI analysis failed: {e}")
            # If analysis fails, still save the file but return error in analysis
            analysis_result = {
                "prediction_class": "Analysis Failed",
                "confidence": 0.0,
                "probabilities": {},
                "class_index": -1,
                "error": str(e)
            }
            xai_url = None
        
        # Store MRI scan metadata in database
        database = await db.get_db()
        scan_doc = {
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "image_url": image_url,
            "xai_url": xai_url,
            "prediction_class": analysis_result.get("prediction_class"),
            "confidence": analysis_result.get("confidence"),
            "probabilities": analysis_result.get("probabilities", {}),
            "class_index": analysis_result.get("class_index"),
            "created_at": datetime.utcnow(),
            "file_size": len(file_content),
            "original_filename": file.filename or "unknown"
        }
        
        await database[db.COL_MRI_SCANS].insert_one(scan_doc)
        
        # Update patient record with latest MRI findings
        await db.patients_update_one(
            {"_id": db.to_object_id(patient_id)},
            {
                "$set": {
                    "imaging_findings": analysis_result.get("prediction_class"),
                    "mri_image_url": image_url,
                    "mri_xai_url": xai_url,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        logger.info(f"MRI scan uploaded and analyzed for patient {patient_id}: {analysis_result.get('prediction_class')}")
        
        return {
            "success": True,
            "image_url": image_url,
            "xai_url": xai_url,
            "prediction_class": analysis_result.get("prediction_class"),
            "confidence": analysis_result.get("confidence"),
            "probabilities": analysis_result.get("probabilities", {}),
            "message": "MRI scan uploaded and analyzed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error processing MRI upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process MRI upload: {str(e)}")


@router.get("/patient/{patient_id}/scans")
async def get_patient_mri_scans(
    patient_id: str,
    doctor=Depends(require_doctor)
):
    """
    Get all MRI scans for a specific patient.
    Returns list of scans with analysis results.
    """
    doctor_id = doctor["sub"]
    
    # Verify patient belongs to this doctor
    patient = await db.patients_find_one({
        "_id": db.to_object_id(patient_id),
        "doctor_id": doctor_id
    })
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found or access denied")
    
    # Get all MRI scans for this patient
    database = await db.get_db()
    cursor = database[db.COL_MRI_SCANS].find(
        {"patient_id": patient_id}
    ).sort("created_at", -1)
    
    scans = []
    async for scan in cursor:
        # Convert ObjectId to string for JSON serialization
        scan["_id"] = str(scan["_id"])
        # Convert datetime to ISO format
        if scan.get("created_at"):
            scan["created_at"] = scan["created_at"].isoformat()
        scans.append(scan)
    
    return {
        "scans": scans,
        "count": len(scans)
    }
