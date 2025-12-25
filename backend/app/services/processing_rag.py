"""
Enhanced Medical RAG Pipeline for Alzheimer's Disease.
Focuses on chunking, embeddings, entity/relationship extraction, and retrieval.
Now uses ChromaDB for vector storage (Local, No Docker).
"""

import os
import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict
import shutil


try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

# PDF reading
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# Embeddings
from sentence_transformers import SentenceTransformer

# PyMuPDF for better title extraction
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# ... (imports continue)

def extract_pdf_title(fp: Path) -> Optional[str]:
    """
    Extract a human-readable title from PDF.
    Priority 1: PyMuPDF (looking for largest font on page 1).
    Priority 2: PDF Metadata.
    Priority 3: First meaningful line.
    """
    title_candidate = None
    
    # Method 1: PyMuPDF (Font Size Analysis)
    if fitz:
        try:
            doc = fitz.open(fp)
            if len(doc) > 0:
                page = doc[0]
                blocks = page.get_text("dict")["blocks"]
                
                max_size = 0
                best_text = ""
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        for span in line["spans"]:
                            # Check for reasonable text
                            text = span["text"].strip()
                            if not text or len(text) < 5:
                                continue
                            
                            # Font size heuristic
                            if span["size"] > max_size:
                                max_size = span["size"]
                                best_text = text
                            elif span["size"] == max_size:
                                # Append if same size (multi-line title)
                                best_text += " " + text
                
                if best_text and max_size > 14: # Assuming titles are larger than normal text (~10-12)
                    # Clean up
                    best_text = clean_text(best_text)
                    if 5 < len(best_text) < 250:
                        return best_text
        except Exception as e:
            print(f"PyMuPDF extraction failed: {e}")

    # Method 2: pypdf Metadata & Heuristics (Playback)
    if PdfReader:
        try:
            reader = PdfReader(str(fp))
            meta = getattr(reader, "metadata", None) or {}
            
            # Metadata Check
            if isinstance(meta, dict):
                meta_title = meta.get("/Title") or meta.get("Title")
            else:
                meta_title = getattr(meta, "title", None)
                
            if meta_title:
                cleaned = str(meta_title).strip()
                bad_titles = ["untitled", "microsoft word", "document", "presentation", "slide", "pdf"]
                if cleaned and 5 < len(cleaned) < 200 and not any(b in cleaned.lower() for b in bad_titles):
                    return cleaned

            # Fallback text analysis (if PyMuPDF wasn't used/failed)
            if reader.pages:
                first_text = (reader.pages[0].extract_text() or "").strip()
                if first_text:
                    lines = first_text.splitlines()
                    junk_patterns = [
                        r'^(https?://|www\.)', r'^\d+$', r'^[A-Z]{2,}\s*\d+', 
                        r'^(doi|DOI|arxiv|ISSN)', r'^(page|Page)\s+\d+', r'^\d{4}-\d{4}', 
                        r'^[©®™]', r'^(Received|Accepted|Published|Available online)', 
                        r'^(Journal|Volume|Issue|Vol\.|No\.)', r'^[A-Z][a-z]+ \d{4}', 
                        r'^Copyright', r'^Elsevier', r'^Springer'
                    ]
                    
                    for line in lines[:20]:
                        line = line.strip()
                        if len(line) < 5: continue
                        if any(re.match(p, line, re.IGNORECASE) for p in junk_patterns): continue
                        if 10 <= len(line) <= 150 and re.search(r"[A-Za-z]", line):
                            if "@" in line or line.endswith("."): continue
                            if line.isupper() or re.search(r"[A-Z][a-z]", line):
                                return line
        except Exception:
            pass
            
    return None
from typing import Literal

# ChromaDB vector database
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False
    print("  chromadb not available. Install: pip install chromadb")

# HuggingFace biomedical NER (lightweight alternative to spaCy)
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    HF_NER_AVAILABLE = True
except Exception:
    HF_NER_AVAILABLE = False
    print("  transformers not available. Install: pip install transformers")


# ----------------------
# Enhanced Config
# ----------------------
MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
MAX_CHARS = 1200
OVERLAP = 250
BATCH_SIZE = 32 
TOPK_SEARCH = 40
TOPK_RERANK = 10 
MIN_RELEVANCE = 0.55  

# ChromaDB Configuration
CHROMA_DB_DIR = "data/chroma_db"
COLLECTION_NAME = "medical_documents"

# ----------------------
# Medical Entity Patterns (Enhanced & Expanded)
# ----------------------
MEDICAL_PATTERNS: Dict[str, List[str]] = {
    "disease": [
        r"\b(alzheimer'?s?|dementia|mild cognitive impairment|mci|frontotemporal|lewy body|vascular dementia)\b",
        r"\b(neurodegeneration|cognitive decline|memory impairment|neurodegenerative disease)\b",
        r"\b(parkinson'?s?|huntington'?s?|creutzfeldt-jakob|progressive supranuclear palsy)\b"
    ],
    "biomarker": [
        r"\b(amyloid|a[βb]eta|aβ42|aβ40|tau|p-tau|t-tau|neurofilament|nfl)\b",
        r"\b(apoe4|apoe|cerebrospinal fluid|csf|plasma biomarker|blood biomarker)\b",
        r"\b(pet scan|fdg-pet|amyloid pet|tau pet|neuroimaging)\b",
        r"\b(phosphorylated tau|hyperphosphorylated tau|total tau)\b"
    ],
    "protein": [
        r"\b(beta-amyloid|amyloid-beta|oligomers|plaques|senile plaques)\b",
        r"\b(neurofibrillary tangles|nfts|tau tangles|paired helical filaments)\b",
        r"\b(synuclein|alpha-synuclein|tdp-43|prion protein)\b"
    ],
    "gene": [
        r"\b(apoe|apoe4|apoe ε4|app|psen1|psen2|presenilin)\b",
        r"\b(trem2|sorl1|cd33|cr1|clusterin|clu)\b",
        r"\b(genetic mutation|familial alzheimer|sporadic alzheimer)\b"
    ],
    "brain_region": [
        r"\b(hippocampus|entorhinal cortex|temporal lobe|frontal lobe|parietal lobe)\b",
        r"\b(amygdala|cerebral cortex|white matter|gray matter|basal ganglia)\b",
        r"\b(prefrontal cortex|occipital lobe|cerebellum|brainstem)\b",
        r"\b(medial temporal lobe|posterior cingulate|precuneus)\b"
    ],
    "symptom": [
        r"\b(memory loss|forgetfulness|confusion|disorientation|amnesia)\b",
        r"\b(apathy|agitation|depression|anxiety|sleep disturbance|insomnia)\b",
        r"\b(language difficulty|aphasia|apraxia|agnosia|dysphagia)\b",
        r"\b(behavioral change|personality change|wandering|sundowning)\b",
        r"\b(executive dysfunction|impaired judgment|difficulty planning)\b",
        r"\b(Balance problems|Tremors)\b"
    ],
    "treatment": [
        r"\b(donepezil|aricept|rivastigmine|exelon|galantamine|razadyne)\b",
        r"\b(memantine|namenda|lecanemab|aducanumab|donanemab|gantenerumab)\b",
        r"\b(cholinesterase inhibitor|nmda antagonist|monoclonal antibody)\b",
        r"\b(cognitive training|rehabilitation|therapy|occupational therapy)\b",
        r"\b(pharmacological|non-pharmacological|disease-modifying|symptomatic treatment)\b"
    ],
    "risk_factor": [
        r"\b(hypertension|diabetes|obesity|smoking|alcohol|hyperlipidemia)\b",
        r"\b(head trauma|tbi|traumatic brain injury|cardiovascular disease|stroke)\b",
        r"\b(genetic risk|family history|apoe4 carrier|hereditary)\b",
        r"\b(age|aging|education level|cognitive reserve|brain reserve)\b",
        r"\b(sedentary lifestyle|social isolation|hearing loss|air pollution)\b"
    ],
    "protective_factor": [
        r"\b(cognitive reserve|brain reserve|education|bilingualism)\b",
        r"\b(physical exercise|mental stimulation|social engagement)\b",
        r"\b(healthy diet|mediterranean diet|mind diet)\b"
    ],
    "assessment": [
        r"\b(mmse|mini-mental|moca|montreal cognitive|cdr|clinical dementia rating)\b",
        r"\b(adas-cog|neuropsychological test|cognitive assessment|memory test)\b",
        r"\b(mri|magnetic resonance|ct scan|brain imaging|structural imaging)\b",
        r"\b(functional assessment|adl|activities of daily living|iadl)\b"
    ],
    "stage": [
        r"\b(preclinical|prodromal|mild|moderate|severe|advanced|terminal)\b",
        r"\b(early-onset|late-onset|young-onset|familial|sporadic)\b",
        r"\b(asymptomatic|symptomatic|progressive|degenerative)\b"
    ],
    "lifestyle": [
        r"\b(exercise|physical activity|aerobic|resistance training|walking)\b",
        r"\b(diet|mediterranean diet|nutrition|omega-3|antioxidants)\b",
        r"\b(social engagement|cognitive stimulation|mental activity|brain training)\b",
        r"\b(sleep quality|sleep hygiene|circadian rhythm|sleep apnea)\b"
    ],
    "mechanism": [
        r"\b(neuroinflammation|oxidative stress|mitochondrial dysfunction)\b",
        r"\b(synaptic loss|neuronal death|cell death|apoptosis)\b",
        r"\b(blood-brain barrier|bbb|neurotransmitter|acetylcholine)\b"
    ]
}

# Enhanced relation patterns
RELATION_PATTERNS: List[Tuple[str, str, str]] = [
    # (pattern, relation_type, confidence)
    (r"(\w[\w\s-]+)\s+(?:is\s+)?associated with\s+(\w[\w\s-]+)", "associated_with", 0.7),
    (r"(\w[\w\s-]+)\s+correlates with\s+(\w[\w\s-]+)", "correlates_with", 0.7),
    (r"(\w[\w\s-]+)\s+(?:can\s+)?treat(?:s)?\s+(\w[\w\s-]+)", "treats", 0.8),
    (r"(\w[\w\s-]+)\s+(?:can\s+)?improve(?:s)?\s+(\w[\w\s-]+)", "improves", 0.8),
    (r"(\w[\w\s-]+)\s+(?:can\s+)?reduce(?:s)?\s+(\w[\w\s-]+)", "reduces", 0.8),
    (r"(\w[\w\s-]+)\s+(?:may\s+)?cause(?:s)?\s+(\w[\w\s-]+)", "causes", 0.6),
    (r"(\w[\w\s-]+)\s+increase(?:s)?\s+(?:the\s+)?risk\s+of\s+(\w[\w\s-]+)", "increases_risk", 0.7),
    (r"(\w[\w\s-]+)\s+prevent(?:s)?\s+(\w[\w\s-]+)", "prevents", 0.8),
    (r"(\w[\w\s-]+)\s+(?:is\s+)?indicated for\s+(\w[\w\s-]+)", "indicated_for", 0.9),
]

# ----------------------
# Utilities
# ----------------------

def read_pdf_text(fp: Path) -> Tuple[str, int]:
    """Extract text from PDF"""
    if PdfReader is None:
        raise RuntimeError("pypdf not installed. Install: pip install pypdf")
    reader = PdfReader(str(fp))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages), len(pages)


def extract_pdf_title(fp: Path) -> Optional[str]:
    """Extract a human-readable title from PDF metadata or first page."""
    if PdfReader is None:
        return None
    try:
        reader = PdfReader(str(fp))
        meta = getattr(reader, "metadata", None) or {}
        title = None
        if isinstance(meta, dict):
            title = meta.get("/Title") or meta.get("Title")
        else:
            title = getattr(meta, "title", None)
            
        # 1. Trust Metadata if it looks good
        if title:
            cleaned = str(title).strip()
            # Filter out common default/bad titles
            bad_titles = ["untitled", "microsoft word", "document", "presentation", "slide", "pdf"]
            if cleaned and 5 < len(cleaned) < 200 and not any(b in cleaned.lower() for b in bad_titles):
                return cleaned

        # 2. Extract from first page text
        if reader.pages:
            first_text = (reader.pages[0].extract_text() or "").strip()
            if first_text:
                lines = first_text.splitlines()
                
                # Enhanced Junk Patterns (Header noise to skip)
                junk_patterns = [
                    r'^(https?://|www\.)',       # URLs
                    r'^\d+$',                    # Just numbers
                    r'^[A-Z]{2,}\s*\d+',         # Document codes "PMC123"
                    r'^(doi|DOI|arxiv|ISSN)',    # Identifiers
                    r'^(page|Page)\s+\d+',       # Page numbers
                    r'^\d{4}-\d{4}',             # Date ranges/ISSNs
                    r'^[©®™]',                   # Copyright
                    r'^(Received|Accepted|Published|Available online)', # Manuscript status
                    r'^(Journal|Volume|Issue|Vol\.|No\.)', # Journal headers
                    r'^[A-Z][a-z]+ \d{4}',       # Date-like lines e.g. "January 2024"
                    r'^Copyright',
                    r'^Elsevier',
                    r'^Springer',
                ]
                
                candidates = []
                
                for line in lines[:20]: # Check first 20 lines only
                    line = line.strip()
                    if not line or len(line) < 5:
                        continue
                        
                    # Skip junk
                    if any(re.match(pattern, line, re.IGNORECASE) for pattern in junk_patterns):
                        continue
                    
                    # Heuristics for a Good Title:
                    # 1. Length: 10-150 chars
                    # 2. Content: mostly letters
                    # 3. Logic: Doesn't end with a period (usually)
                    # 4. Filter: Not an email list
                    
                    if 10 <= len(line) <= 150 and re.search(r"[A-Za-z]", line):
                        if "@" in line: # Skip email lines
                            continue
                        if line.endswith("."): # Titles rarely end in periods
                            continue
                            
                        # High priority: All CAPS or Title Case
                        score = 0
                        if line.isupper(): score += 2
                        if re.search(r"[A-Z][a-z]", line): score += 1
                        
                        candidates.append((score, line))
                
                # Return best candidate
                if candidates:
                    # Sort by score desc, then position asc (prefer earlier lines if tie)
                    # Actually stable sort by score is enough if we reverse
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    return candidates[0][1]
                    
                # Last resort fallback: First non-empty line
                for line in lines:
                    if len(line.strip()) > 10:
                        return line.strip()

    except Exception:
        return None
    return None


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove emails and URLs
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"doi:\s*\S+", "", text, flags=re.I)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove page numbers and headers/footers patterns
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    
    return text


def semantic_chunk_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> List[str]:
    """
    Semantic chunking - split at paragraph boundaries when possible
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks: List[str] = []
    
    # Split by double newlines (paragraphs)
    paragraphs = text.split("\n\n")
    
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If single paragraph is too long, split it
        if len(para) > max_chars:
            # Save current chunk if exists
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split long paragraph with overlap
            step = max_chars - overlap
            for i in range(0, len(para), step):
                chunk = para[i:i + max_chars]
                if chunk:
                    chunks.append(chunk)
            continue
        
        # Try to add paragraph to current chunk
        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk += para + "\n\n"
        else:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def stable_id(doc_name: str, chunk_index: int) -> str:
    """Generate stable chunk ID"""
    return f"{doc_name}::{chunk_index}"


# ----------------------
# ChromaDB Client Management
# ----------------------

_CHROMA_CLIENT = None
_CHROMA_COLLECTION = None

def get_chroma_client(db_path: str = CHROMA_DB_DIR):
    """Get or create cached ChromaDB client instance"""
    global _CHROMA_CLIENT
    
    if _CHROMA_CLIENT is not None:
        return _CHROMA_CLIENT
    
    if not CHROMA_AVAILABLE:
        print(" ChromaDB not available. Please install: pip install chromadb")
        return None
    
    try:
        # Resolve absolute path for db_dir relative to project root
        # Assuming this script is in backend/app/services/
        # Project root is backend/
        root_dir = Path(__file__).parent.parent.parent
        abs_db_path = str(root_dir / db_path)
        
        print(f"Connecting to ChromaDB at {abs_db_path}...")
        client = chromadb.PersistentClient(path=abs_db_path)
        _CHROMA_CLIENT = client
        return client
    except Exception as e:
        print(f" Failed to connect to ChromaDB: {e}")
        return None

def get_chroma_collection(client, collection_name: str = COLLECTION_NAME):
    """Get or create ChromaDB collection"""
    global _CHROMA_COLLECTION
    
    if _CHROMA_COLLECTION is not None:
        return _CHROMA_COLLECTION

    if client is None:
        return None
        
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )
        _CHROMA_COLLECTION = collection
        return collection
    except Exception as e:
        print(f"Failed to get/create Chroma collection: {e}")
        return None


# ----------------------
# Embedding
# ----------------------

# Global cache for embedder (performance optimization)
_EMBEDDER_CACHE = {}

def get_embedder(model_name: str = MODEL):
    """Get or create cached embedder instance"""
    global _EMBEDDER_CACHE
    
    if model_name in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[model_name]
    
    print(f"Loading embedding model: {model_name}")
    embedder = Embedder(model_name)
    _EMBEDDER_CACHE[model_name] = embedder
    return embedder

class Embedder:
    def __init__(self, model_name: str = MODEL):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.model_name = model_name
        print(f"Model loaded (dimension: {self.dim})")

    def embed_texts(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """Batch embed texts"""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed single query"""
        vec = self.model.encode([text], show_progress_bar=False, normalize_embeddings=True)[0]
        return vec.astype(np.float32)


# ----------------------
# Medical Entity Extraction
# ----------------------

def extract_medical_entities_regex(text: str) -> Dict[str, Any]:
    """Extract medical entities using regex patterns"""
    results: Dict[str, Any] = {"counts": defaultdict(int), "entities": defaultdict(set)}
    
    text_lower = text.lower()
    
    for entity_type, patterns in MEDICAL_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                results["counts"][entity_type] += len(matches)
                results["entities"][entity_type].update(matches)
    
    # Convert sets to lists for JSON serialization
    results["entities"] = {k: list(v) for k, v in results["entities"].items()}
    results["counts"] = dict(results["counts"])
    
    return results


# Global cache for HF NER pipeline (performance optimization)
_HF_NER_CACHE = None

def build_hf_ner_pipeline(model_name: str = "d4data/biomedical-ner-all"):
    """Build HuggingFace NER pipeline with caching for performance"""
    global _HF_NER_CACHE
    
    # Return cached pipeline if available
    if _HF_NER_CACHE is not None:
        return _HF_NER_CACHE
    
    if not HF_NER_AVAILABLE:
        return None
    
    try:
        print(f"Loading HuggingFace NER model: {model_name}...")
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForTokenClassification.from_pretrained(model_name)
        _HF_NER_CACHE = pipeline(
            "token-classification",
            model=mdl,
            tokenizer=tok,
            aggregation_strategy="simple"
        )
        print("HuggingFace NER model loaded successfully.")
        return _HF_NER_CACHE
    except Exception as e:
        print(f"Failed to load HuggingFace NER: {e}")
        return None


def extract_medical_entities_hf(text: str, hf_ner_pipeline, max_len: int = 5000) -> Dict[str, Any]:
    """Extract medical entities from single text using HuggingFace NER"""
    if hf_ner_pipeline is None:
        return {"counts": {}, "entities": {}}
    sample = text[:max_len]
    try:
        ents = hf_ner_pipeline(sample)
        by_type = defaultdict(set)
        for e in ents:
            et = e.get("entity_group", "OTHER").lower()
            term = e.get("word", "").lower().strip()
            # Clean up subword tokens (##)
            term = term.replace("##", "")
            if len(term) >= 3:
                by_type[et].add(term)
        return {
            "counts": {k: len(v) for k, v in by_type.items()},
            "entities": {k: list(v) for k, v in by_type.items()}
        }
    except Exception:
        return {"counts": {}, "entities": {}}


def boost_score_by_entities(base_score: float, entity_data: Dict[str, Any]) -> float:
    """Boost relevance score based on medical entity density"""
    total_entities = sum(entity_data.get("counts", {}).values())
    
    # More aggressive boost for medical content
    boost = min(0.35, 0.025 * total_entities)
    
    return min(1.0, base_score + boost)


# ----------------------
# Relation Extraction
# ----------------------

def extract_medical_relations(text: str, max_len: int = 5000) -> List[Dict[str, Any]]:
    """
    Extract medical relationships from text
    """
    relations: List[Dict[str, Any]] = []
    text_sample = text[:max_len]
    
    for pattern, rel_type, confidence in RELATION_PATTERNS:
        for match in re.finditer(pattern, text_sample, re.IGNORECASE):
            groups = match.groups()
            if len(groups) >= 2:
                source = groups[0].strip().lower()
                target = groups[1].strip().lower()
                
                # Filter out very short or generic terms
                if len(source) < 3 or len(target) < 3:
                    continue
                if source in ["the", "a", "an", "this", "that"] or target in ["the", "a", "an", "this", "that"]:
                    continue
                
                relations.append({
                    "source": source,
                    "target": target,
                    "relation": rel_type,
                    "confidence": confidence,
                    "context": match.group(0)
                })
    
    return relations


# ----------------------
# Enhanced Knowledge Graph
# ----------------------

class MedicalKnowledgeGraph:
    """
    Enhanced knowledge graph for Alzheimer's domain
    """
    def __init__(self):
        self.entities: Dict[str, Set[str]] = defaultdict(set)
        self.relations: List[Dict[str, Any]] = []
        self.entity_to_docs: Dict[str, Set[str]] = defaultdict(set)
        
    def add_entities(self, entity_data: Dict[str, Any], doc_name: str):
        """Add entities from a document"""
        for entity_type, entity_list in entity_data.get("entities", {}).items():
            for entity in entity_list:
                self.entities[entity_type].add(entity)
                self.entity_to_docs[entity].add(doc_name)
    
    def add_relations(self, relations: List[Dict[str, Any]], doc_name: str):
        """Add relations from a document"""
        for rel in relations:
            rel_copy = dict(rel)
            rel_copy["document"] = doc_name
            self.relations.append(rel_copy)
    
    def get_entity_stats(self) -> Dict[str, int]:
        """Get entity statistics"""
        return {etype: len(entities) for etype, entities in self.entities.items()}
    
    def find_related_entities(self, entity: str, relation_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find entities related to a given entity"""
        related = []
        entity_lower = entity.lower()
        
        for rel in self.relations:
            if relation_type and rel["relation"] != relation_type:
                continue
            
            if entity_lower in rel["source"].lower():
                related.append({"target": rel["target"], "relation": rel["relation"], "confidence": rel["confidence"]})
            elif entity_lower in rel["target"].lower():
                related.append({"source": rel["source"], "relation": rel["relation"], "confidence": rel["confidence"]})
        
        return related
    
    def reason_about_patient(self, patient_ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced patient reasoning using knowledge graph
        """
        symptoms = [s.lower() for s in patient_ctx.get("symptoms", [])]
        mmse = patient_ctx.get("mmse_score")
        moca = patient_ctx.get("moca_score")
        biomarkers = patient_ctx.get("biomarkers", {})
        
        indicators = []
        relevant_treatments = set()
        relevant_assessments = set()
        
        # Symptom-based reasoning
        cognitive_symptoms = ["memory loss", "confusion", "forgetfulness", "disorientation"]
        behavioral_symptoms = ["agitation", "apathy", "depression", "anxiety"]
        
        if any(s in symptoms for s in cognitive_symptoms):
            indicators.append({"symptom": "cognitive_decline", "indicates": "mild", "confidence": 0.7})
        
        if any(s in symptoms for s in behavioral_symptoms):
            indicators.append({"symptom": "behavioral_changes", "indicates": "moderate", "confidence": 0.6})
        
        # MMSE-based reasoning
        if mmse is not None:
            if mmse >= 24:
                indicators.append({"assessment": "mmse", "indicates": "normal", "confidence": 0.9})
            elif mmse >= 20:
                indicators.append({"assessment": "mmse", "indicates": "mild", "confidence": 0.9})
            elif mmse >= 10:
                indicators.append({"assessment": "mmse", "indicates": "moderate", "confidence": 0.9})
            else:
                indicators.append({"assessment": "mmse", "indicates": "severe", "confidence": 0.9})
        
        # MoCA-based reasoning
        if moca is not None:
            if moca < 26:
                indicators.append({"assessment": "moca", "indicates": "impairment", "confidence": 0.8})
        
        # Biomarker-based reasoning
        if biomarkers.get("amyloid") == "positive":
            indicators.append({"biomarker": "amyloid", "indicates": "alzheimer_pathology", "confidence": 0.85})
        if biomarkers.get("tau") == "elevated":
            indicators.append({"biomarker": "tau", "indicates": "neurodegeneration", "confidence": 0.85})
        
        # Find relevant treatments from knowledge graph
        for symptom in symptoms:
            related = self.find_related_entities(symptom, relation_type="treats")
            for item in related:
                if "source" in item:
                    relevant_treatments.add(item["source"])
        
        return {
            "stage_indicators": indicators,
            "suggested_treatments": list(relevant_treatments),
            "suggested_assessments": list(relevant_assessments),
            "total_indicators": len(indicators)
        }
    
    def export_graph(self, output_path: Path):
        """Export knowledge graph to JSON"""
        # Convert sets to lists for JSON serialization
        entities_json = {k: list(v) for k, v in self.entities.items()}
        # Convert entity_to_docs sets to lists
        entity_to_docs_json = {k: list(v) for k, v in self.entity_to_docs.items()}
        
        graph_data = {
            "entities": entities_json,
            "relations": self.relations,
            "entity_to_docs": entity_to_docs_json,
            "entity_stats": self.get_entity_stats(),
            "total_relations": len(self.relations)
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Knowledge graph exported to {output_path}")


# ----------------------
# MMR Reranking
# ----------------------

def mmr_rerank(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    lambda_param: float = 0.6,
    top_k: int = TOPK_RERANK
) -> List[int]:
    """
    Maximal Marginal Relevance reranking for diversity
    """
    if doc_vecs.size == 0 or len(doc_vecs) == 0:
        return []
    
    # Normalize vectors
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    d_norm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9)
    
    # Compute similarity to query
    similarities = d_norm @ q_norm
    
    selected_indices: List[int] = []
    remaining_indices = list(range(len(doc_vecs)))
    
    # Select first document (highest similarity)
    if remaining_indices:
        first_idx = int(np.argmax(similarities))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
    
    # Iteratively select diverse documents
    while len(selected_indices) < min(top_k, len(doc_vecs)) and remaining_indices:
        # Compute diversity (max similarity to already selected)
        selected_vecs = d_norm[selected_indices]
        remaining_vecs = d_norm[remaining_indices]
        
        diversity = np.max(selected_vecs @ remaining_vecs.T, axis=0)
        
        # MMR score: balance relevance and diversity
        mmr_scores = lambda_param * similarities[remaining_indices] - (1 - lambda_param) * diversity
        
        # Select document with highest MMR score
        best_local_idx = int(np.argmax(mmr_scores))
        best_idx = remaining_indices[best_local_idx]
        
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    return selected_indices


# ----------------------
# Document Processing
# ----------------------

def process_documents(
    pdf_dir: Path,
    out_dir: Path,
    model_name: str = MODEL,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP,
    batch_size: int = BATCH_SIZE,
    ner_mode: Literal["regex", "hf"] = "hf",
    limit: Optional[int] = None
) -> Tuple[Path, Path]:
    """
    Process PDF documents into embeddings and knowledge graph.
    Stores vectors in ChromaDB and exports KG to JSON.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]
    
    print(f"Found {len(pdfs)} PDFs in {pdf_dir}")
    print(f"Chunk size: {max_chars}, overlap: {overlap}")
    print(f"NER mode: {ner_mode}")
    print(f"Using ChromaDB at {CHROMA_DB_DIR}")
    
    embedder = get_embedder(model_name)
    kg = MedicalKnowledgeGraph()
    
    # Initialize ChromaDB
    chroma_client = get_chroma_client()
    if chroma_client is None:
        raise RuntimeError("ChromaDB unavailable. Cannot proceed.")
        
    # Reset/Create collection
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass # Collection might not exist
        
    collection = get_chroma_collection(chroma_client, COLLECTION_NAME)
    
    hf_ner = None
    if ner_mode == "hf":
        hf_ner = build_hf_ner_pipeline()
        if hf_ner is None:
            print("HuggingFace NER unavailable; falling back to regex")
            ner_mode = "regex"

    total_chunks = 0
    buffer_nodes = []
    buffer_embeddings = []
    
    for pdf_path in tqdm(pdfs, desc="Processing PDFs"):
        try:
            # Extract text
            text, num_pages = read_pdf_text(pdf_path)
            text = clean_text(text)
            # Extract a better document title
            doc_title = extract_pdf_title(pdf_path)
            if not doc_title:
                doc_title = pdf_path.stem.replace("_", " ").replace("-", " ")
            
            # Semantic chunking
            chunks = semantic_chunk_text(text, max_chars, overlap)
            
            # Embed chunks
            chunk_embeddings = embedder.embed_texts(chunks, batch_size=batch_size)
            
            # Process each chunk
            for idx, chunk in enumerate(chunks):
                # Extract medical entities
                if ner_mode == "hf" and hf_ner is not None:
                    entities_hf = extract_medical_entities_hf(chunk, hf_ner)
                    entities_regex = extract_medical_entities_regex(chunk)
                    # Merge HF and regex results
                    merged = {
                        "counts": defaultdict(int),
                        "entities": defaultdict(set)
                    }
                    for et, c in entities_regex.get("counts", {}).items():
                        merged["counts"][et] += c
                    for et, ents in entities_regex.get("entities", {}).items():
                        merged["entities"][et].update(ents)
                    for et, c in entities_hf.get("counts", {}).items():
                        merged["counts"][et] += c
                    for et, ents in entities_hf.get("entities", {}).items():
                        merged["entities"][et].update(ents)
                    entities = {
                        "counts": dict(merged["counts"]),
                        "entities": {k: list(v) for k, v in merged["entities"].items()}
                    }
                else:
                    # Fallback to regex only
                    entities = extract_medical_entities_regex(chunk)
                
                # Extract relations (only from first few chunks to save time)
                relations = []
                if idx < 5:  # Top 5 chunks per document
                    relations = extract_medical_relations(chunk)
                    kg.add_relations(relations, pdf_path.name)
                
                # Add entities to knowledge graph
                kg.add_entities(entities, pdf_path.name)
                
                # Calculate relevance score
                base_relevance = 0.5
                relevance = boost_score_by_entities(base_relevance, entities)
                
                # Prepare node data for Chroma
                node_id = stable_id(pdf_path.name, idx)
                metadata = {
                    "document_name": pdf_path.name,
                    "document_title": doc_title,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "page_number": 0,
                    "total_pages": num_pages,
                    "content_type": "text",
                    "relevance_score": float(relevance),
                    "entity_counts": json.dumps(entities.get("counts", {})), # flatten for Chroma
                    "has_relations": len(relations) > 0
                }
                
                buffer_nodes.append({
                    "id": node_id,
                    "text": chunk,
                    "metadata": metadata
                })
                buffer_embeddings.append(chunk_embeddings[idx])
                
                # Flush buffer if full
                if len(buffer_nodes) >= BATCH_SIZE:
                    collection.add(
                        ids=[n["id"] for n in buffer_nodes],
                        documents=[n["text"] for n in buffer_nodes],
                        metadatas=[n["metadata"] for n in buffer_nodes],
                        embeddings=[e.tolist() for e in buffer_embeddings]
                    )
                    total_chunks += len(buffer_nodes)
                    buffer_nodes = []
                    buffer_embeddings = []
        
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            continue
    
    # Flush remaining buffer
    if buffer_nodes:
        collection.add(
            ids=[n["id"] for n in buffer_nodes],
            documents=[n["text"] for n in buffer_nodes],
            metadatas=[n["metadata"] for n in buffer_nodes],
            embeddings=[e.tolist() for e in buffer_embeddings]
        )
        total_chunks += len(buffer_nodes)

    print(f"✅ Stored {total_chunks} chunks in ChromaDB")
    
    # Save artifacts (Just KG and Stats)
    kg_path = out_dir / "knowledge_graph.json"
    stats_path = out_dir / "processing_stats.json"
    
    # Export knowledge graph
    kg.export_graph(kg_path)
    
    # Save processing stats
    stats = {
        "total_documents": len(pdfs),
        "total_chunks": total_chunks,
        "embedding_dimension": embedder.dim,
        "chunk_size": max_chars,
        "overlap": overlap,
        "entity_stats": kg.get_entity_stats(),
        "total_relations": len(kg.relations),
        "storage": "ChromaDB"
    }
    
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    print("Processing complete.")
    print(f"Knowledge Graph: {kg_path}")
    print(f"Stats: {stats_path}")
    
    return kg_path, stats_path


# ----------------------
# Retrieval
# ----------------------

def retrieve(
    out_dir: Path,
    query: str,
    model_name: str = MODEL,
    topk_search: int = TOPK_SEARCH,
    topk_rerank: int = TOPK_RERANK,
    min_relevance: float = MIN_RELEVANCE,
    patient_ctx: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks with MMR reranking using ChromaDB.
    """
    print(f"Query: {query}")
    
    # Connect to Chroma
    client = get_chroma_client()
    if client is None:
        print("❌ ChromaDB unavailable.")
        return []
        
    collection = get_chroma_collection(client)
    
    # Embed query
    embedder = get_embedder(model_name)
    query_vec = embedder.embed_query(query)
    
    # Load knowledge graph for boosting
    kg_path = out_dir / "knowledge_graph.json"
    kg = MedicalKnowledgeGraph()
    if kg_path.exists():
        with open(kg_path, "r", encoding="utf-8") as f:
            kg_data = json.load(f)
            kg.relations = kg_data.get("relations", [])
            for etype, ents in kg_data.get("entities", {}).items():
                kg.entities[etype] = set(ents)
    
    print("🔍 Searching with ChromaDB...")
    
    # Query Chroma
    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=topk_search,
        include=["metadatas", "documents", "distances", "embeddings"]
    )
    
    # Process results
    candidates = []
    candidate_vecs = []
    
    if not results['ids'] or not results['ids'][0]:
        print("⚠️  No results found in ChromaDB")
        return []

    num_hits = len(results['ids'][0])
    
    for i in range(num_hits):
        dist = results['distances'][0][i]
        similarity = 1 - dist # Chroma returns distance
        
        if similarity >= min_relevance:
            metadata = results['metadatas'][0][i]
            # Parse entity_counts back from JSON string if needed
            if isinstance(metadata.get("entity_counts"), str):
                try:
                    metadata["entity_counts"] = json.loads(metadata["entity_counts"])
                except:
                    pass
            
            candidates.append({
                "text": results['documents'][0][i],
                "score": float(similarity),
                "metadata": metadata,
                "node_id": results['ids'][0][i]
            })
            
            # Store vector for MMR reranking is tricky without API returning it
            # But we added "embeddings" to include list
            if results.get('embeddings') is not None:
                candidate_vecs.append(results['embeddings'][0][i])
    
    if len(candidates) == 0:
        print("⚠️  No results above relevance threshold")
        return []
    
    # MMR reranking for diversity
    if len(candidate_vecs) > 0:
        candidate_vecs_np = np.array(candidate_vecs, dtype=np.float32)
        reranked_indices = mmr_rerank(query_vec, candidate_vecs_np, lambda_param=0.6, top_k=min(topk_rerank, len(candidates)))
        final_results = [candidates[i] for i in reranked_indices]
    else:
        final_results = candidates[:topk_rerank]
    
    
    # Knowledge graph boost (if patient context provided)
    if patient_ctx:
        kg_insights = kg.reason_about_patient(patient_ctx)
        indicated_stages = [ind.get("indicates") for ind in kg_insights.get("stage_indicators", [])]
        
        if indicated_stages:
            boosted_results = []
            for result in final_results:
                text_lower = result["text"].lower()
                boosted = dict(result)
                
                for stage in indicated_stages:
                    if stage and stage.lower() in text_lower:
                        boosted["score"] *= 1.15  # 15% boost
                        boosted["kg_boosted"] = True
                        break
                
                boosted_results.append(boosted)
            
            # Re-sort by boosted scores
            boosted_results.sort(key=lambda x: x["score"], reverse=True)
            final_results = boosted_results
            
            print(f"Knowledge graph indicators: {len(kg_insights.get('stage_indicators', []))}")
    
    print(f"Retrieved {len(final_results)} results (from {len(candidates)} candidates)")
    
    return final_results


# ----------------------
# Simple CLI
# ----------------------

def main():
    """Simple command-line interface.
    Commands: process | test | "your query". Use --ner=regex|hf (default: hf).
    """
    import sys
    
    # Auto-detect paths
    script_dir = Path(__file__).parent.parent.parent  # backend/
    pdf_dir = script_dir / "data" / "docs"
    out_dir = script_dir / "data" / "processed_enhanced"
    
    # Get command and optional args
    command = sys.argv[1] if len(sys.argv) > 1 else "process"
    # Simple arg parsing for ner mode
    ner_mode = "hf"  # Default to HuggingFace NER
    model_override = None
    topk_rerank_override = None
    min_rel_override = None
    for arg in sys.argv[2:]:
        if arg.startswith("--ner="):
            val = arg.split("=", 1)[1].strip().lower()
            if val in {"regex", "hf"}:
                ner_mode = val
        elif arg.startswith("--model="):
            model_override = arg.split("=", 1)[1].strip()
        elif arg.startswith("--topk_rerank="):
            try:
                topk_rerank_override = int(arg.split("=", 1)[1].strip())
            except Exception:
                pass
        elif arg.startswith("--min_relevance="):
            try:
                min_rel_override = float(arg.split("=", 1)[1].strip())
            except Exception:
                pass
    
    # Parse limit
    limit_val = None
    for arg in sys.argv:
        if arg.startswith("--limit="):
            try:
                limit_val = int(arg.split("=", 1)[1])
            except:
                pass

    if command == "process" or command == "reprocess":
        print("Processing documents...")
        print(f"PDF Directory: {pdf_dir}")
        print(f"Output Directory: {out_dir}")
        print(f"Chunk Size: {MAX_CHARS}")
        print(f"Overlap: {OVERLAP}")
        print(f"Storage: ChromaDB ({CHROMA_DB_DIR})")
        if limit_val:
            print(f"Limit: {limit_val} documents")
        
        if not pdf_dir.exists():
            print(f"❌ Error: PDF directory not found: {pdf_dir}")
            print(f"   Please create it and add your PDF files")
            return
        
        process_documents(
            pdf_dir=pdf_dir,
            out_dir=out_dir,
            model_name=(model_override or MODEL),
            max_chars=MAX_CHARS,
            overlap=OVERLAP,
            batch_size=BATCH_SIZE,
            ner_mode=ner_mode,
            limit=limit_val
        )
        
        print("Done. To test retrieval: python app/services/processing_rag.py test")
    
    elif command == "test":
        print("Testing retrieval...")
        
        # Test queries
        test_queries = [
            "What are the early symptoms of Alzheimer's disease?",
            "What treatments are available for Alzheimer's?",
            "What are the risk factors for dementia?",
            "How is Alzheimer's diagnosed?",
            "What lifestyle changes can help prevent Alzheimer's?"
        ]
        
        print("Running test queries...\n")
        
        for i, query in enumerate(test_queries, 1):
            print(f"Query {i}: {query}")
            
            results = retrieve(
                out_dir=out_dir,
                query=query,
                model_name=(model_override or MODEL),
                topk_search=TOPK_SEARCH,
                topk_rerank=(topk_rerank_override or 5),
                min_relevance=(min_rel_override or MIN_RELEVANCE),
                patient_ctx=None
            )
            
            if results:
                for j, result in enumerate(results[:3], 1):  # Show top 3
                    meta = result.get("metadata", {})
                    print(f"\n  [{j}] Score: {result['score']:.3f}")
                    print(f"      Doc: {meta.get('document_name', 'Unknown')[:50]}")
                    print(f"      Preview: {result['text'][:150]}...")
            else:
                print("  No results found")
        
        print("Testing complete.")
    
    else:
        # Treat as a query
        query = " ".join(sys.argv[1:])
        
        print(f"Searching: {query}")
        
        results = retrieve(
            out_dir=out_dir,
            query=query,
            model_name=(model_override or MODEL),
            topk_search=TOPK_SEARCH,
            topk_rerank=(topk_rerank_override or TOPK_RERANK),
            min_relevance=(min_rel_override or MIN_RELEVANCE),
            patient_ctx=None
        )
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                meta = result.get("metadata", {})
                print(f"[{i}] Score: {result['score']:.3f}")
                print(f"    Doc: {meta.get('document_name', 'Unknown')}")
                print(f"    Chunk: {meta.get('chunk_index', 0)}/{meta.get('total_chunks', 0)}")
                print(f"    Entities: {meta.get('entity_counts', {})}")
                if result.get("kg_boosted"):
                    print(f"    Knowledge graph boosted")
                print(f"    Text: {result['text'][:200]}...")
                print()
        else:
            print("No results found")
        
        print("Done.")


if __name__ == "__main__":
    main()
