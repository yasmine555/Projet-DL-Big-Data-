from fastapi import FastAPI
from app.routers import rag
import os

# defer heavy imports until startup to avoid import-time model loading
app = FastAPI(title="Medical RAG Assistant")

app.include_router(rag.router, prefix="/rag", tags=["RAG"])


@app.on_event("startup")
def startup_event():

    from app.config import OUT_DIR, PDF_DIR
    from app.services.alzheimer_rag import (
        
        AlzheimerRetriever,
        AlzheimerRAGSystem,
        RAGConfig,
    )




@app.get("/")
def root():
    return {"message": "Medical AI Assistant Local RAG Backend is running."}
