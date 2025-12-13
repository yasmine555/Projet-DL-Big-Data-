"""
Enhanced Medical RAG Pipeline for Alzheimer's Disease.
Focuses on chunking, embeddings, entity/relationship extraction, and retrieval.
"""

import os
import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict

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
from typing import Literal

# Weaviate vector database
try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    from weaviate.classes.query import MetadataQuery
    WEAVIATE_AVAILABLE = True
except Exception:
    WEAVIATE_AVAILABLE = False
    print("  weaviate-client not available. Install: pip install weaviate-client")

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
MIN_RELEVANCE = 0.55  # 

# Weaviate Configuration
WEAVIATE_URL = "http://localhost:8081"  # Local Weaviate instance (port 8081 to avoid conflict)
WEAVIATE_COLLECTION = "MedicalDocuments"

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
    """Extract a human-readable title from PDF metadata or first page.

    Priority:
    1) PDF metadata title
    2) First non-empty line from first page that looks like a title (<= 200 chars)
    """
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
        if title:
            cleaned = str(title).strip()
            if cleaned and cleaned.lower() not in ["untitled", "microsoft word", "document"]:
                return cleaned

        # Extract from first page
        if reader.pages:
            first_text = (reader.pages[0].extract_text() or "").strip()
            if first_text:
                # Skip common junk patterns
                junk_patterns = [
                    r'^(https?://|www\.)',  # URLs
                    r'^\d+$',  # Just numbers
                    r'^[A-Z]{2,}\s*\d+',  # Document codes like "PMC123456"
                    r'^(doi|DOI|arxiv)',  # DOI/arxiv prefixes
                    r'^(page|Page)\s+\d+',  # Page numbers
                    r'^\d{4}-\d{4}',  # ISSNs
                    r'^[©®™]',  # Copyright symbols
                ]
                
                for line in first_text.splitlines():
                    line = line.strip()
                    if not line or len(line) < 10:
                        continue
                    
                    # Skip junk patterns
                    if any(re.match(pattern, line, re.IGNORECASE) for pattern in junk_patterns):
                        continue
                    
                    # Look for title-like lines (10-200 chars, contains letters)
                    if 10 <= len(line) <= 200 and re.search(r"[A-Za-z]", line):
                        # Prefer lines with title case or sentence case
                        if re.search(r"[A-Z][a-z]", line):
                            return line
                
                # Fallback: return first reasonable line
                for line in first_text.splitlines():
                    line = line.strip()
                    if 10 <= len(line) <= 200 and re.search(r"[A-Za-z]", line):
                        return line
                        
                # Last resort: first 120 chars
                return first_text[:120]
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
# Weaviate Client Management
# ----------------------

# Global cache for Weaviate client
_WEAVIATE_CLIENT = None

def get_weaviate_client(url: str = WEAVIATE_URL):
    """Get or create cached Weaviate client instance"""
    global _WEAVIATE_CLIENT
    
    if _WEAVIATE_CLIENT is not None:
        return _WEAVIATE_CLIENT
    
    if not WEAVIATE_AVAILABLE:
        print(" Weaviate client not available. Please install: pip install weaviate-client")
        return None
    
    try:
        print(f"Connecting to Weaviate at {url}...")
        client = weaviate.connect_to_local(host="localhost", port=8081)
        
        if client.is_ready():
            print(" Connected to Weaviate successfully")
            _WEAVIATE_CLIENT = client
            return client
        else:
            print(" Weaviate is not ready. Make sure Docker container is running.")
            return None
    except Exception as e:
        print(f" Failed to connect to Weaviate: {e}")
        print("   Make sure Weaviate is running: docker-compose -f docker-compose.weaviate.yml up -d")
        return None

def create_weaviate_schema(client):
    """Create Weaviate schema for medical documents"""
    if client is None:
        return False
    
    try:
        # Check if collection already exists
        if client.collections.exists(WEAVIATE_COLLECTION):
            print(f"Collection '{WEAVIATE_COLLECTION}' already exists")
            return True
        
        # Create collection with properties
        client.collections.create(
            name=WEAVIATE_COLLECTION,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="document_name", data_type=DataType.TEXT),
                Property(name="document_title", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="total_chunks", data_type=DataType.INT),
                Property(name="page_number", data_type=DataType.INT),
                Property(name="total_pages", data_type=DataType.INT),
                Property(name="relevance_score", data_type=DataType.NUMBER),
                Property(name="entity_counts", data_type=DataType.TEXT),  # JSON string
                Property(name="has_relations", data_type=DataType.BOOL),
            ],
            # We'll provide our own vectors (from SentenceTransformer)
            vectorizer_config=Configure.Vectorizer.none(),
        )
        
        print(f" Created collection '{WEAVIATE_COLLECTION}'")
        return True
    except Exception as e:
        print(f" Failed to create schema: {e}")
        return False


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


def extract_medical_entities_hf_batch(texts: List[str], hf_ner_pipeline, max_len: int = 5000) -> List[Dict[str, Any]]:
    """Batch extract medical entities from multiple texts (performance optimization)"""
    if hf_ner_pipeline is None:
        return [{"counts": {}, "entities": {}} for _ in texts]
    
    results = []
    for text in texts:
        result = extract_medical_entities_hf(text, hf_ner_pipeline, max_len)
        results.append(result)
    
    return results


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
        graph_data = {
            "entities": {k: list(v) for k, v in self.entities.items()},
            "relations": self.relations,
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
) -> Tuple[Path, Path, Path, Path]:
    """
    Process PDF documents into embeddings and knowledge graph.
    
    Args:
        ner_mode: 'hf' for HuggingFace biomedical NER (recommended), 'regex' for pattern matching only
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]
    
    print(f"Found {len(pdfs)} PDFs in {pdf_dir}")
    print(f"Chunk size: {max_chars}, overlap: {overlap}")
    print(f"NER mode: {ner_mode}")
    
    embedder = get_embedder(model_name)
    kg = MedicalKnowledgeGraph()
    
    all_nodes: List[Dict[str, Any]] = []
    all_embeddings: List[np.ndarray] = []
    
    hf_ner = None
    if ner_mode == "hf":
        hf_ner = build_hf_ner_pipeline()
        if hf_ner is None:
            print("HuggingFace NER unavailable; falling back to regex")
            ner_mode = "regex"

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
            all_embeddings.append(chunk_embeddings)
            
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
                
                # Create node
                node = {
                    "id": stable_id(pdf_path.name, idx),
                    "text": chunk,
                    "metadata": {
                        "document_name": pdf_path.name,
                        "document_title": doc_title,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "page_number": 0,  # Could be enhanced to track actual pages
                        "total_pages": num_pages,
                        "content_type": "text",
                        "relevance_score": float(relevance),
                        "entity_counts": entities.get("counts", {}),
                        "has_relations": len(relations) > 0
                    }
                }
                
                all_nodes.append(node)
        
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            continue
    
    
    # Connect to Weaviate and create schema
    weaviate_client = get_weaviate_client()
    use_weaviate = weaviate_client is not None
    
    if use_weaviate:
        create_weaviate_schema(weaviate_client)
        collection = weaviate_client.collections.get(WEAVIATE_COLLECTION)
        
        # Store all chunks in Weaviate with batch processing
        print(f"Storing {len(all_nodes)} chunks in Weaviate...")
        with collection.batch.dynamic() as batch:
            for idx, node in enumerate(tqdm(all_nodes, desc="Uploading to Weaviate")):
                # Get corresponding embedding
                embedding = all_embeddings[idx // len(all_embeddings[0])][idx % len(all_embeddings[0])]
                
                # Prepare properties
                properties = {
                    "text": node["text"],
                    "document_name": node["metadata"]["document_name"],
                    "document_title": node["metadata"]["document_title"],
                    "chunk_index": node["metadata"]["chunk_index"],
                    "total_chunks": node["metadata"]["total_chunks"],
                    "page_number": node["metadata"]["page_number"],
                    "total_pages": node["metadata"]["total_pages"],
                    "relevance_score": node["metadata"]["relevance_score"],
                    "entity_counts": json.dumps(node["metadata"]["entity_counts"]),
                    "has_relations": node["metadata"]["has_relations"],
                }
                
                # Add object with vector
                batch.add_object(
                    properties=properties,
                    vector=embedding.tolist()
                )
        
        print(f"✅ Stored {len(all_nodes)} chunks in Weaviate")
    
    # Save artifacts (backup/fallback)
    nodes_path = out_dir / "nodes.json"
    embeddings_path = out_dir / "embeddings.npy"
    kg_path = out_dir / "knowledge_graph.json"
    stats_path = out_dir / "processing_stats.json"
    
    # Save nodes (backup)
    with open(nodes_path, "w", encoding="utf-8") as f:
        json.dump(all_nodes, f, indent=2)
    
    # Save embeddings (backup)
    if all_embeddings:
        embeddings_matrix = np.vstack(all_embeddings)
        np.save(embeddings_path, embeddings_matrix)
    
    # Export knowledge graph
    kg.export_graph(kg_path)
    
    # Save processing stats
    stats = {
        "total_documents": len(pdfs),
        "total_chunks": len(all_nodes),
        "avg_chunks_per_doc": len(all_nodes) / len(pdfs) if pdfs else 0,
        "embedding_dimension": embedder.dim,
        "chunk_size": max_chars,
        "overlap": overlap,
        "entity_stats": kg.get_entity_stats(),
        "total_relations": len(kg.relations),
        "using_weaviate": use_weaviate
    }
    
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    
    print("Processing complete.")
    if use_weaviate:
        print(f"✅ Weaviate: {len(all_nodes)} chunks stored")
    print(f"Nodes (backup): {nodes_path}")
    print(f"Embeddings (backup): {embeddings_path}")
    print(f"Knowledge Graph: {kg_path}")
    print(f"Stats: {stats_path}")
    print("Summary:")
    print(f"  Documents: {len(pdfs)}")
    print(f"  Chunks: {len(all_nodes)}")
    print(f"  Entities: {sum(kg.get_entity_stats().values())}")
    print(f"  Relations: {len(kg.relations)}")
    
    return nodes_path, embeddings_path, kg_path, stats_path


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
    Retrieve relevant chunks with MMR reranking.
    Uses Weaviate if available, falls back to NumPy files.
    """
    print(f"Query: {query}")
    
    # Try Weaviate first
    weaviate_client = get_weaviate_client()
    use_weaviate = weaviate_client is not None and weaviate_client.collections.exists(WEAVIATE_COLLECTION)
    
    # Embed query
    embedder = get_embedder(model_name)
    query_vec = embedder.embed_query(query)
    
    # Load knowledge graph
    kg_path = out_dir / "knowledge_graph.json"
    kg = MedicalKnowledgeGraph()
    if kg_path.exists():
        with open(kg_path, "r", encoding="utf-8") as f:
            kg_data = json.load(f)
            kg.relations = kg_data.get("relations", [])
            for etype, ents in kg_data.get("entities", {}).items():
                kg.entities[etype] = set(ents)
    
    if use_weaviate:
        print("🔍 Searching with Weaviate...")
        collection = weaviate_client.collections.get(WEAVIATE_COLLECTION)
        
        # Query Weaviate with vector search
        response = collection.query.near_vector(
            near_vector=query_vec.tolist(),
            limit=topk_search,
            return_metadata=MetadataQuery(distance=True)
        )
        
        # Convert Weaviate results to our format
        candidates = []
        candidate_vecs = []
        
        for obj in response.objects:
            # Calculate similarity from distance (Weaviate returns cosine distance)
            similarity = 1 - obj.metadata.distance
            
            if similarity >= min_relevance:
                # Parse entity_counts back from JSON
                entity_counts = json.loads(obj.properties.get("entity_counts", "{}")) if obj.properties.get("entity_counts") else {}
                
                candidates.append({
                    "text": obj.properties["text"],
                    "score": float(similarity),
                    "metadata": {
                        "document_name": obj.properties.get("document_name", ""),
                        "document_title": obj.properties.get("document_title", ""),
                        "chunk_index": obj.properties.get("chunk_index", 0),
                        "total_chunks": obj.properties.get("total_chunks", 0),
                        "page_number": obj.properties.get("page_number", 0),
                        "total_pages": obj.properties.get("total_pages", 0),
                        "relevance_score": obj.properties.get("relevance_score", 0.0),
                        "entity_counts": entity_counts,
                        "has_relations": obj.properties.get("has_relations", False),
                    },
                    "node_id": str(obj.uuid)
                })
                
                # Store vector for MMR reranking
                if obj.vector is not None:
                    candidate_vecs.append(obj.vector)
        
        if len(candidates) == 0:
            print("⚠️  No results above relevance threshold")
            return []
        
        # MMR reranking for diversity
        if len(candidate_vecs) > 0:
            candidate_vecs_np = np.array(candidate_vecs, dtype=np.float32)
            reranked_indices = mmr_rerank(query_vec, candidate_vecs_np, lambda_param=0.6, top_k=min(topk_rerank, len(candidates)))
            results = [candidates[i] for i in reranked_indices]
        else:
            results = candidates[:topk_rerank]
        
    else:
        # Fallback to NumPy files
        print("⚠️  Weaviate unavailable, using file-based retrieval...")
        nodes_path = out_dir / "nodes.json"
        embeddings_path = out_dir / "embeddings.npy"
        
        if not nodes_path.exists() or not embeddings_path.exists():
            print("❌ No processed data found. Run document processing first.")
            return []
        
        with open(nodes_path, "r", encoding="utf-8") as f:
            nodes = json.load(f)
        
        embeddings = np.load(embeddings_path)
        
        # Compute similarities
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        doc_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        similarities = doc_norms @ query_norm
        
        # Get top-k candidates
        top_indices = np.argsort(similarities)[::-1][:topk_search]
        top_sims = similarities[top_indices]
        
        # Filter by minimum relevance
        valid_mask = top_sims >= min_relevance
        top_indices = top_indices[valid_mask]
        top_sims = top_sims[valid_mask]
        
        if len(top_indices) == 0:
            print("⚠️  No results above relevance threshold")
            return []
        
        # MMR reranking for diversity
        candidate_vecs = embeddings[top_indices]
        reranked_local_indices = mmr_rerank(query_vec, candidate_vecs, lambda_param=0.6, top_k=topk_rerank)
        reranked_indices = [top_indices[i] for i in reranked_local_indices]
        
        # Build results
        results = []
        for idx in reranked_indices:
            node = nodes[idx]
            score = float(similarities[idx])
            
            results.append({
                "text": node["text"],
                "score": score,
                "metadata": node.get("metadata", {}),
                "node_id": node.get("id", "")
            })
    
    # Knowledge graph boost (if patient context provided)
    if patient_ctx:
        kg_insights = kg.reason_about_patient(patient_ctx)
        indicated_stages = [ind.get("indicates") for ind in kg_insights.get("stage_indicators", [])]
        
        if indicated_stages:
            boosted_results = []
            for result in results:
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
            results = boosted_results
            
            print(f"Knowledge graph indicators: {len(kg_insights.get('stage_indicators', []))}")
    
    print(f"Retrieved {len(results)} results (from {len(top_indices)} candidates)")
    
    return results


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
    
    if command == "process" or command == "reprocess":
        print("Processing documents...")
        print(f"PDF Directory: {pdf_dir}")
        print(f"Output Directory: {out_dir}")
        print(f"Chunk Size: {MAX_CHARS}")
        print(f"Overlap: {OVERLAP}")
        
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
            limit=None
        )
        
        print("Done. To test retrieval: python enhanced_medical_rag.py test")
    
    elif command == "test":
        print("Testing retrieval...")
        
        if not out_dir.exists():
            print(f"❌ Error: Processed data not found: {out_dir}")
            print(f"   Run 'python {Path(__file__).name}' first to process documents")
            return
        
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
        
        if not out_dir.exists():
            print(f" Error: Processed data not found: {out_dir}")
            print(f"   Run 'python {Path(__file__).name}' first to process documents")
            return
        
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
