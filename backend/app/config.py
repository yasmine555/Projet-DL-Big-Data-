import http
import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = os.getenv("PDF_DIR", str(BASE_DIR / "data" / "docs"))
OUT_DIR = os.getenv("OUT_DIR", str(BASE_DIR / "data" / "processed"))
# Optional: use separate test artifacts without touching production
USE_TEST_ARTIFACTS = os.getenv("USE_TEST_ARTIFACTS", "false").lower() in ("1", "true", "yes")
TEST_OUT_DIR = os.getenv("TEST_OUT_DIR", str(BASE_DIR / "data" / "processed_test"))
ACTIVE_OUT_DIR = TEST_OUT_DIR if USE_TEST_ARTIFACTS else OUT_DIR
CHUNK_DIR = os.path.join(OUT_DIR, "chunks")

# Fixed: Define INDEX_PATH before using it
INDEX_PATH = os.path.join(ACTIVE_OUT_DIR, "embeddings.npy")
NODES_PATH = os.path.join(ACTIVE_OUT_DIR, "nodes.json")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TEST_OUT_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
os.makedirs(os.path.dirname(NODES_PATH), exist_ok=True)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE", 100))
EMBEDDING_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEN_MODEL_NAME ="openai/gpt-oss-120b"
TOP_K_RETRIEVAL=15
MAX_TOKENS=1024
MRI_MODEL_PATH = os.getenv("MRI_MODEL_PATH", str(BASE_DIR /"app"/ "models" / "best_model.h5"))
IMAGE_PATH= os.getenv("IMAGE_PATH", str(BASE_DIR / "data" / "mri_image"))
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
        out_dir: str = ACTIVE_OUT_DIR
        chunk_dir: str = CHUNK_DIR
        index_path: str = INDEX_PATH
        nodes_path: str = NODES_PATH
        chunk_size: int = CHUNK_SIZE
        overlap_size: int = OVERLAP_SIZE
        embedding_model_name: str = EMBEDDING_MODEL_NAME
        device: str = str(DEVICE)
        gen_model_name: str = GEN_MODEL_NAME
        top_k_retrieval: int = TOP_K_RETRIEVAL
        max_tokens: int = MAX_TOKENS

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
        out_dir = ACTIVE_OUT_DIR
        chunk_dir = CHUNK_DIR
        index_path = INDEX_PATH
        nodes_path = NODES_PATH
        chunk_size = CHUNK_SIZE
        overlap_size = OVERLAP_SIZE
        embedding_model_name = EMBEDDING_MODEL_NAME
        device = str(DEVICE)
        gen_model_name = GEN_MODEL_NAME
        top_k_retrieval = TOP_K_RETRIEVAL
        max_tokens = MAX_TOKENS
        # runtime
        ENVIRONMENT = ENVIRONMENT
        HOST = HOST
        PORT = PORT
        DEBUG = DEBUG

    settings = _SimpleSettings()
N8N_WEBHOOK_URL="http://localhost:5678/webhook/doctor-events"
