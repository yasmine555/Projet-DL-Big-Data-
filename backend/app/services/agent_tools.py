import json
import os
from bson import ObjectId

import networkx as nx
from typing import Optional
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
import requests

from app.services.processing_rag import (
    get_chroma_collection,
    get_embedder,
    get_chroma_client
)
from app.models.db import COL_PATIENTS, COL_RESULTS
from app.config import KG_PATH
MONGO_URL = os.getenv("MONGO_URI")
MONGO_DB= os.getenv("MONGO_DB")

# ---------------------------
# MongoDB (lazy init)
# ---------------------------
_MONGO_CLIENT = None

def get_mongo_db():
    global _MONGO_CLIENT
    if _MONGO_CLIENT is None:
        _MONGO_CLIENT = AsyncIOMotorClient(MONGO_URL or "mongodb://localhost:27017/")
    return _MONGO_CLIENT[MONGO_DB or "alz_rag"]
async def get_patient_records(patient_id: str) -> str:
    """Retrieve patient medical record and latest MRI results."""
    db = get_mongo_db()
    print(f"DEBUG: get_patient_records called with ID: '{patient_id}'")
    
    patient = None
    try:
        # Try querying by _id (ObjectId)
        patient = await db[COL_PATIENTS].find_one({"_id": ObjectId(patient_id)})
    except Exception as e:
        print(f"DEBUG: Error converting '{patient_id}' to ObjectId: {e}")
        # Fallback: try querying by patient_id string field if _id fails
        patient = await db[COL_PATIENTS].find_one({"patient_id": patient_id})

    if not patient:
        print(f"DEBUG: Patient '{patient_id}' NOT FOUND in DB. Checking for orphaned results...")
        patient = {} # Use empty dict to avoid AttributeError later

    # Fetch latest MRI result
    latest_result = await db[COL_RESULTS].find_one(
        {"patient_id": patient_id},
        sort=[("created_at", -1)]
    )

    if not patient and not latest_result:
         print(f"DEBUG: Neither patient nor result found for '{patient_id}'")
         return "No patient record found."

    # Helper to serialize Mongo documents (handle ObjectId, datetime)
    def _serialize_doc(doc):
        if not doc:
            return {}
        out = {}
        for k, v in doc.items():
            if isinstance(v, ObjectId):
                out[k] = str(v)
            elif hasattr(v, "isoformat"):
                out[k] = v.isoformat()
            else:
                out[k] = v
        return out

    # Serialize full documents to ensure NO data is hidden from agent
    patient_dump = _serialize_doc(patient)
    result_dump = _serialize_doc(latest_result)

    summary = {
        "source": "database_dump",
        "description": "Full patient details and latest medical results.",
        "patient_record": patient_dump,
        "latest_analysis_result": result_dump
    }

    return json.dumps(summary, indent=2)

# ---------------------------
# Medical Document Search
# ---------------------------
def search_medical_docs(query: str) -> str:
    """Semantic search over medical documents."""
    client = get_chroma_client()
    collection = get_chroma_collection(client)
    embedder = get_embedder()

    print(f"DEBUG: search_medical_docs called with query: '{query}'")
    query_emb = embedder.embed_query(query)

    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=5
    )

    if not results["documents"] or not results["documents"][0]:
        print("DEBUG: No documents found in Chroma.")
        return f"No relevant medical documents found in local database for query: {query}"

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    
    print(f"DEBUG: Found {len(docs)} documents.")

    out = []
    for doc, meta in zip(docs, metas):
        title = meta.get("document_title", "Unknown")
        print(f"DEBUG: Retrieved doc title: {title}")
        out.append(f"Source: {title}\nContent: {doc}")

    return "\n---\n".join(out)

# ---------------------------
# Knowledge Graph Tool
# ---------------------------
_GRAPH_CACHE = None

def load_graph():
    global _GRAPH_CACHE
    if _GRAPH_CACHE is None:
        path = Path(KG_PATH)
        print(f"DEBUG: Loading Knowledge Graph from {path}")
        if not path.exists():
            print("DEBUG: KG file does not exist!")
            return nx.Graph()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        G = nx.Graph()
        for rel in data.get("relations", []):
            G.add_edge(rel["source"], rel["target"], relation=rel.get("relation"))
        _GRAPH_CACHE = G
        print(f"DEBUG: Graph loaded with {len(G.nodes)} nodes and {len(G.edges)} edges.")

    return _GRAPH_CACHE

def query_knowledge_graph(entity: str) -> str:
    """Query relations for a medical entity."""
    print(f"DEBUG: query_knowledge_graph called for entity: '{entity}'")
    G = load_graph()

    node = next((n for n in G.nodes if n.lower() == entity.lower()), None)
    if not node:
        print(f"DEBUG: '{entity}' NOT found in graph.")
        return f"{entity} not found in knowledge graph."
    
    print(f"DEBUG: Found node '{node}'")

    lines = []
    for u, v, data in G.edges(node, data=True):
        other = v if u == node else u
        lines.append(f"{node} --[{data.get('relation','related_to')}]--> {other}")

    return "\n".join(lines[:15])

# ---------------------------
# Free Scientific Research Tools
# ---------------------------

async def fetch_latest_research(query: str) -> str:
    """
    Fetch the latest scientific research papers from PubMed and Semantic Scholar.
    Use this for clinical trials, latest dementia studies, and medical journal findings.
    """
    print(f"DEBUG: fetch_latest_research called for: '{query}'")
    
    results = []
    
    # 1. PubMed API (NIH) - No key required for basic usage
    try:
        # Search PubMed
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmode=json&retmax=3"
        s_res = requests.get(search_url, timeout=5).json()
        id_list = s_res.get("esearchresult", {}).get("idlist", [])
        
        if id_list:
            ids = ",".join(id_list)
            sum_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={ids}&retmode=json"
            summ = requests.get(sum_url, timeout=5).json()
            
            for pid in id_list:
                doc = summ.get("result", {}).get(pid, {})
                title = doc.get("title", "Unknown Title")
                date = doc.get("pubdate", "Unknown Date")
                results.append(f"### [PubMed] {title}\nDate: {date}\nLink: https://pubmed.ncbi.nlm.nih.gov/{pid}/")
    except Exception as e:
        print(f"DEBUG: PubMed search failed: {e}")

    # 2. Semantic Scholar API - Free Public Access
    try:
        ss_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=3&fields=title,url,abstract,year,authors"
        ss_res = requests.get(ss_url, timeout=5).json()
        
        for paper in ss_res.get("data", []):
            title = paper.get("title", "Unknown Title")
            year = paper.get("year", "N/A")
            url = paper.get("url", "#")
            abstract = paper.get("abstract", "")
            if abstract:
                abstract = abstract[:300] + "..."
            
            results.append(f"### [Semantic Scholar] {title}\nYear: {year}\nAbstract: {abstract}\nLink: {url}")
    except Exception as e:
        print(f"DEBUG: Semantic Scholar search failed: {e}")

    if not results:
        return f"No live research papers found for query: {query}"

    return "\n\n---\n\n".join(results)


def web_search_n8n(query: str) -> str:
    """
    Performs a real-time web search using an external n8n webhook.
    Use this for very recent news, drug approvals, or information not in the local database.
    
    Args:
        query: The search query string.
    """
    webhook = os.getenv("N8N_SEARCH_WEBHOOK")
    if not webhook:
        return "Web search not configured."

    r = requests.post(webhook, json={"query": query}, timeout=10)
    return r.text if r.status_code == 200 else "Search failed."
