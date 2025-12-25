import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = os.getenv("PDF_DIR", str(BASE_DIR / "data" / "docs"))
OUT_DIR = os.getenv("OUT_DIR", str(BASE_DIR / "data" / "processed_enhanced"))



os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
UPLOAD_DIR=os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))
KG_PATH=os.getenv("KG_PATH", str(BASE_DIR / "data" / "processed_enhanced" / "knowledge_graph.json"))
CHROMA_DB=os.getenv("CHROMA_DB", str(BASE_DIR / "data" / "chroma_db"))
EMBEDDING_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEN_MODEL_NAME ="openai/gpt-oss-120b"

CLASSIFICATION_MODEL = os.getenv("MRI_MODEL_PATH", str(BASE_DIR /"app"/ "models" / "best_model.h5"))
# Directory to store uploaded MRI images (served via /uploads)
MRI_UPLOAD_DIR = os.getenv("MRI_UPLOAD_DIR", str(BASE_DIR / "app" / "uploads" / "mri"))
os.makedirs(MRI_UPLOAD_DIR, exist_ok=True)
# Settings object for backward compatibility
try:
    from pydantic import BaseSettings
except Exception:
    BaseSettings = None

# runtime defaults from env
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")

if BaseSettings is not None:
    class Settings(BaseSettings):
        base_dir: str = str(BASE_DIR)
        pdf_dir: str = PDF_DIR
        out_dir: str = OUT_DIR

        embedding_model_name: str = EMBEDDING_MODEL_NAME
        device: str = str(DEVICE)
        gen_model_name: str = GEN_MODEL_NAME


        # runtime
        ENVIRONMENT: str = ENVIRONMENT
        HOST: str = HOST
        PORT: int = PORT
        DEBUG: bool = DEBUG

        class Config:
            env_prefix = ""
            env_file = ".env"

    settings = Settings()
else:
    class _SimpleSettings:
        base_dir = str(BASE_DIR)
        pdf_dir = PDF_DIR
        out_dir = OUT_DIR
        embedding_model_name = EMBEDDING_MODEL_NAME
        device = str(DEVICE)
        gen_model_name = GEN_MODEL_NAME

        # runtime
        ENVIRONMENT = ENVIRONMENT
        HOST = HOST
        PORT = PORT
        DEBUG = DEBUG

    settings = _SimpleSettings()

N8N_WEBHOOK_URL="http://localhost:5678/webhook/doctor-events"
