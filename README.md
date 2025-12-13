# Agentic Alzheimer Detection Platform

An end-to-end platform combining deep learning on brain MRI for Alzheimer detection, explainable AI (XAI) to interpret model decisions, and an agentic RAG system for evidence-grounded clinical assistance. Frontend (Vite/React) + Backend (FastAPI) with data pipelines and notebooks for visualization.

## Features
- **Deep Learning MRI Analysis**: CNN-based classifier trained on Alzheimer MRI datasets; supports transfer learning and preprocessing of 2D slices.
- **XAI for Trust**: SHAP/CAPTUM visualizations and saliency maps highlighting regions influencing predictions.
- **Agentic RAG System**: Tool-driven retrieval using biomedical embeddings, entity/relations extraction, MMR reranking, and role-aware answers (patient vs doctor) with citations.
- **Knowledge Graph**: Extracted medical entities and relations; interactive visualization via PyVis and static NetworkX views.
- **Role-Aware Orchestration**: Summarizer → internal search → optional external links → evaluator → formatter for safe, grounded responses.

## Datasets Overview (data/)
This project uses multimodal datasets for neurodegenerative diseases. Primary focus for DL classification is Alzheimer MRI.

- **Alzheimer’s**: OASIS and Kaggle 4-class MRI datasets (`.nii`, `.jpg`).


> Large raw data and processed artifacts are ignored in Git. See `.gitignore`. Provide paths as described below.

## Backend Setup
```powershell
Push-Location "c:\Bureau\projet_Dl\Projet-DL-Big-Data-\backend"; python -m venv venv; ./venv/Scripts/activate.ps1; pip install -r requirements.txt; Pop-Location
```

Environment variables (create `backend/.env` or root `.env`):
- `GROQ_API_KEY` or provider keys if used for LLM


Run API (FastAPI via Uvicorn):
```powershell
Push-Location "c:\Bureau\projet_Dl\Projet-DL-Big-Data-\backend\app";pip install reauirements.txt; ./../venv/Scripts/activate.ps1; uvicorn main:app --reload; Pop-Location
```
## Frontend Setup
```powershell
Push-Location "c:\Bureau\projet_Dl\Projet-DL-Big-Data-\frontend"; npm install; npm run dev; Pop-Location
```

Environment (`frontend/.env`):
- `VITE_API_BASE_URL=http://localhost:8000`

# Original Dataset Details (Reference)
This directory contains datasets used in the project "Explainable AI System for the Diagnosis of Neurodegenerative Diseases".
The datasets cover Alzheimer’s Disease, Parkinson’s Disease, and Multiple Sclerosis (MS), using multimodal data such as MRI scans, tabular biomarkers, and voice metadata.

🧩 Dataset Overview
Disease	Data Type	Source	Format	Folder
Alzheimer’s Disease	MRI (structural brain images)	OASIS
 & Alzheimer’s Multiclass Kaggle Dataset
	.nii, .jpg	/data/Alzheimer/
🧬 1. Alzheimer’s Disease Datasets
🧠 A. OASIS MRI Dataset



