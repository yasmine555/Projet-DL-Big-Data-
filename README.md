README – Neurodegenerative Diseases Datasets
Directory: data/

This directory contains all datasets used in the project "Explainable AI System for the Diagnosis of Neurodegenerative Diseases".
The datasets cover Alzheimer’s Disease, Parkinson’s Disease, and Multiple Sclerosis (MS), using multimodal data such as MRI scans, tabular biomarkers, and voice metadata.

🧩 Dataset Overview
Disease	Data Type	Source	Format	Folder
Alzheimer’s Disease	MRI (structural brain images)	OASIS
 & Alzheimer’s Multiclass Kaggle Dataset
	.nii, .jpg	/data/Alzheimer/
Parkinson’s Disease	MRI + Voice features (metadata)	NTUA Parkinson MRI Dataset
 + MDVR-KCL Voice Dataset
	.jpg, .csv, .json	/data/Parkinson/
Multiple Sclerosis (MS)	MRI (lesion-based)	MCND Dataset (Kaggle)
	.jpg	/data/MS/
🧬 1. Alzheimer’s Disease Datasets
🧠 A. OASIS MRI Dataset

Source: Washington University – OASIS-1 Project

Description: 3D brain MRI scans of subjects categorized by the Clinical Dementia Rating (CDR) scale.

Classes:

Non-demented

Very Mild Demented

Mild Demented

Moderate Demented

Format: .nii (NIfTI) volumes converted to 2D slices for CNN training.

Folder:

data/Alzheimer/OASIS/

🧩 B. Alzheimer’s Disease Multiclass Dataset (Kaggle)

Source: Kaggle – Alzheimer’s Dataset (4 Classes)

Description: 44,000 MRI images (JPEG) categorized by disease severity.

Classes:

NonDemented

VeryMildDemented

MildDemented

ModerateDemented

Format: .jpg

Folder:

data/Alzheimer/Alzheimer_4class/

🧠 2. Parkinson’s Disease Datasets
🧩 A. NTUA Parkinson MRI Dataset

Source: Kaggle – NTUA Parkinson Brain MRI Dataset

Description: Brain MRI images used to detect Parkinson’s-related structural patterns.

Labels: Parkinson’s Disease (PD) vs Healthy Control (HC)

Format: .jpg + metadata .csv

Folder:

data/Parkinson/MRI_NTUA/

Example of data.csv
id	diagnosis	radius_mean	texture_mean	perimeter_mean	...
842302	M	17.99	10.38	122.8	...
842517	M	20.57	17.77	132.9	...

The file contains pre-extracted numeric features representing morphological and statistical descriptors from MRI regions of interest (ROIs).

🎙️ B. MDVR-KCL Voice Dataset (JSON/CSV)

Source: Zenodo – MDVR-KCL Dataset

Description:
Voice recordings collected via mobile devices from Parkinson’s and control participants.
Includes metadata with UPDRS scores, Hoehn & Yahr ratings, and health labels.

Format: .json and .csv (metadata and annotations)

Folder:

data/Parkinson/Audio_MDVR/

Example of nde-zenodo_2867215.json
{
  "name": "Mobile Device Voice Recordings at King's College London",
  "description": "Voice data for Parkinson's early detection",
  "labels": ["PD", "HC"],
  "metadata": {
    "sample_rate": 44100,
    "bit_depth": 16,
    "recording_device": "Motorola Moto G4"
  }
}


Although raw .wav files are not publicly accessible, this JSON/CSV provides detailed metadata for analysis and feature-based modeling (e.g., jitter, shimmer, MFCCs).

🧩 3. Multiple Sclerosis (MS) Dataset
🧠 MCND Dataset – MS Subset

Source: Kaggle – MCND Dataset

Description: MRI scans categorized by neurological condition (Alzheimer’s, Brain Tumor, MS, and Healthy).

Subset Used: Only the MS and Normal classes for binary classification.

Format: .jpg

Folder:

data/MS/MCND_MS/

📊 Summary
Disease	Data Type	Format	Classes	Folder
Alzheimer’s	MRI	.nii, .jpg	4 classes (dementia severity)	/data/Alzheimer/
Parkinson’s	MRI + voice (metadata)	.jpg, .csv, .json	PD vs HC	/data/Parkinson/
Multiple Sclerosis	MRI	.jpg	MS vs Normal	/data/MS/

---------------
# Backend Setup
cd backend
python -m venv venv 

venv/Scripts/activate

pip install -r requirements.txt

uvicorn main:app --reload

# Frontend Setup


cd frontend

npm install

npm start


