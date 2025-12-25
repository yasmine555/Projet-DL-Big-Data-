from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question or query text")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional patient/context metadata")
    user_type: Optional[str] = Field("patient", description="patient|doctor")
    explain: Optional[bool] = Field(True, description="Return explanations/evidence")
    patient_id: Optional[str] = Field(None, description="Patient ID (Mongo ObjectId as string) when doctor queries for a specific patient")
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="Conversation history")


class Biomarkers(BaseModel):
    amyloid: Optional[str] = Field("unknown", description="amyloid: unknown/negative/positive")
    tau: Optional[str] = Field("unknown", description="tau: unknown/negative/positive")
    apoe4: Optional[str] = Field("unknown", description="apoe4: unknown/negative/positive")


class QuestionnaireMetadata(BaseModel):
    name: Optional[str] = Field(None, description="Patient Name")
    age: Optional[int] = Field(None)
    sex: Optional[str] = Field(None)
    education_level: Optional[str] = Field(None)
    cognitive_reserve_years: Optional[float] = Field(None)
    family_history: Optional[bool] = Field(False)
    current_medications: Optional[List[str]] = Field(default_factory=list)
    symptoms_list: Optional[List[str]] = Field(default_factory=list)
    medical_history: Optional[List[str]] = Field(default_factory=list)
    user_type: Optional[str] = Field("patient")
    mmse_score: Optional[float] = Field(None)
    moca_score: Optional[float] = Field(None)
    symptom_onset_years: Optional[float] = Field(None)
    biomarkers: Optional[Biomarkers] = Field(default_factory=Biomarkers)
    imaging_findings: Optional[str] = Field(None)
    neuro_exam_notes: Optional[str] = Field(None)
    differential_diagnosis: Optional[str] = Field(None)
    mri_uploaded: Optional[bool] = Field(False)
    extra: Optional[Dict[str, Any]] = Field(None)


class QuestionnaireRequest(BaseModel):
    responses: List[int] = Field(..., description="Numeric vector of questionnaire answers (non-empty)")
    metadata: QuestionnaireMetadata = Field(..., description="Patient / questionnaire metadata")
    patient_id: Optional[str] = Field(None, description="Patient ID (Mongo ObjectId as string) when saving results for a doctor")


class PatientBase(BaseModel):
    name: str
    age: Optional[int] = None
    sex: Optional[str] = Field(default=None, description="male|female|other")
    # New fields
    education_level: Optional[str] = None
    years_of_education: Optional[int] = None
    medical_history: Optional[str] = None
    family_history: Optional[str] = None
    mse: Optional[float] = None
    moca: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="arbitrary clinical metrics")


class PatientCreate(PatientBase):
    pass


class PatientUpdate(PatientBase):
    pass