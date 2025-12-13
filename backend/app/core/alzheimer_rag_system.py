import os
import json
import numpy as np
import pickle
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import torch
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re
from dotenv import load_dotenv
from groq import Groq

from transformers import AutoTokenizer, AutoModel
from llama_index.core.schema import TextNode
from app.config import (GEN_MODEL_NAME, CHUNK_DIR, NODES_PATH, OUT_DIR, 
                        PDF_DIR, DEVICE, EMBEDDING_MODEL_NAME, TOP_K_RETRIEVAL)
from app.services.knowledge_graph import MedicalKnowledgeGraph

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


@dataclass
class RAGConfig:
    """Enhanced RAG configuration"""
    pdf_dir: str = PDF_DIR
    output_dir: str = OUT_DIR
    embeddings_path: str = NODES_PATH  # NODES_PATH should be the JSON path
    embedding_model: str = EMBEDDING_MODEL_NAME
    llm_model: str = GEN_MODEL_NAME
    
    # Retrieval parameters - OPTIMIZED FOR SPEED
    top_k_retrieval: int = 8
    min_relevance_score: float = 0.6
    diversity_threshold: float = 0.85
    rerank_top_k: int = 5
    
    # Generation parameters - PERSONA-SPECIFIC
    patient_temperature: float = 0.25  # Slightly higher for warmth and natural flow
    doctor_temperature: float = 0.1    # Lower for precision and consistency
    temperature: float = 0.2           # Default fallback
    max_tokens: int = 1200
    top_p: float = 0.9                 # Nucleus sampling parameter
    presence_penalty: float = 0.0      # Reduce repetition if needed
    
    # XAI parameters
    enable_reasoning_trace: bool = True
    enable_confidence_scoring: bool = True
    enable_source_validation: bool = True
    min_confidence_threshold: float = 0.6
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class UserType(Enum):
    DOCTOR = "doctor"
    PATIENT = "patient"
    CAREGIVER = "caregiver"


class QueryType(Enum):
    SYMPTOM_INQUIRY = "symptom"
    TREATMENT_OPTIONS = "treatment"
    LIFESTYLE_ADVICE = "lifestyle"
    CAREGIVER_SUPPORT = "caregiver"
    DIAGNOSIS_INFO = "diagnosis"
    RISK_FACTORS = "risk_factors"
    GENERAL = "general"
    GREETING = "greeting"
    GRATITUDE = "gratitude"


class AlzheimerStage(Enum):
    HEALTHY = "healthy"
    MILD_IMPAIRMENT = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


@dataclass
class PatientContext:
    """Patient information for personalized responses"""
    age: Optional[int] = None
    symptoms: List[str] = None
    stage: AlzheimerStage = AlzheimerStage.UNKNOWN
    medical_history: List[str] = None
    current_medications: List[str] = None
    mmse_score: Optional[float] = None
    moca_score: Optional[float] = None
    biomarkers: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.symptoms is None:
            self.symptoms = []
        if self.medical_history is None:
            self.medical_history = []
        if self.current_medications is None:
            self.current_medications = []
        if self.biomarkers is None:
            self.biomarkers = {}
        if self.extra is None:
            self.extra = {}

    def to_context_string(self) -> str:
        """Convert to concise readable string"""
        parts = []
        if self.age:
            parts.append(f"Age {self.age}")
        if self.symptoms:
            parts.append(f"Symptoms: {', '.join(self.symptoms[:4])}")
        if self.mmse_score is not None:
            parts.append(f"MMSE {self.mmse_score}/30")
        if self.moca_score is not None:
            parts.append(f"MoCA {self.moca_score}/30")
        return " | ".join(parts) if parts else ""
    
    def generate_patient_summary(self) -> str:
        """Generate a brief AI summary of patient information - OBSERVATIONS ONLY, NO RECOMMENDATIONS"""
        if not any([self.age, self.symptoms, self.mmse_score, self.moca_score]):
            return "No patient information available."
        
        summary_parts = []
        
        # Age context - neutral observation only
        if self.age:
            if self.age < 60:
                summary_parts.append(f"Patient is {self.age} years old, potentially in early-onset category")
            elif self.age >= 65:
                summary_parts.append(f"Patient is {self.age} years old, in typical age range for Alzheimer's")
            else:
                summary_parts.append(f"Patient is {self.age} years old")
        
        # Cognitive assessment - clinical observations only
        cognitive_status = []
        if self.mmse_score is not None:
            if self.mmse_score >= 24:
                cognitive_status.append(f"MMSE {self.mmse_score}/30 (normal range)")
            elif self.mmse_score >= 20:
                cognitive_status.append(f"MMSE {self.mmse_score}/30 (mild cognitive impairment range)")
            elif self.mmse_score >= 10:
                cognitive_status.append(f"MMSE {self.mmse_score}/30 (moderate impairment range)")
            else:
                cognitive_status.append(f"MMSE {self.mmse_score}/30 (severe impairment range)")
        
        if self.moca_score is not None:
            if self.moca_score >= 26:
                cognitive_status.append(f"MoCA {self.moca_score}/30 (normal range)")
            else:
                cognitive_status.append(f"MoCA {self.moca_score}/30 (below normal threshold)")
        
        if cognitive_status:
            summary_parts.append("; ".join(cognitive_status))
        
        # Symptoms - list only, no interpretation
        if self.symptoms:
            symptom_list = ', '.join(self.symptoms[:3])
            if len(self.symptoms) > 3:
                symptom_list += f" (+{len(self.symptoms) - 3} more)"
            summary_parts.append(f"Reported symptoms: {symptom_list}")
        
        # Stage assessment - factual only
        if self.stage != AlzheimerStage.UNKNOWN:
            summary_parts.append(f"Current stage: {self.stage.value}")
        
        return ". ".join(summary_parts) + "."


@dataclass
class ReasoningStep:
    """Represents a single reasoning step"""
    step_number: int
    action: str
    reasoning: str
    result: str
    confidence: float
    timestamp: str


@dataclass
class ExplainableResult:
    """Complete explainable result with reasoning trace"""
    answer: str
    reasoning_trace: List[ReasoningStep]
    confidence_score: float
    sources: List[Dict]
    retrieval_metrics: Dict
    generation_metrics: Dict
    warnings: List[str]
    recommendations: List[str]
    patient_summary: Optional[str] = None


class EnhancedRetriever:
    """Intelligent retriever with quality filtering + Knowledge Graph - OPTIMIZED"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        print(f"🔧 Initializing Enhanced Retriever with Knowledge Graph...")
        
        # Initialize knowledge graph
        self.kg = MedicalKnowledgeGraph()
        print(f"✓ Knowledge graph initialized")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.embedding_model,
            model_max_length=512
        )
        self.model = AutoModel.from_pretrained(config.embedding_model)
        self.model.eval()
        
        try:
            self.model.to(config.device)
            print(f"✓ Model loaded on {config.device}")
        except Exception:
            print(f"✓ Model loaded on CPU")

        self.nodes = self._load_nodes()
        if self.nodes:
            self.embeddings_matrix = np.vstack([
                np.array(n.embedding, dtype=np.float32) for n in self.nodes
            ])
            print(f"✓ Loaded {len(self.nodes)} document chunks")
        else:
            self.embeddings_matrix = np.empty((0,))
            print(f"⚠ No embeddings found")

    def _load_nodes(self) -> List[TextNode]:
        """Load nodes from JSON and attach embeddings from embeddings.npy"""
        nodes_path = Path(NODES_PATH) 
        emb_path = Path(OUT_DIR) / "embeddings.npy"

        if not nodes_path.exists():
            print(f"⚠ nodes.json not found at {nodes_path}")
            return []
        if not emb_path.exists():
            print(f"⚠ embeddings.npy not found at {emb_path}")
            return []

        try:
            with open(nodes_path, "r", encoding="utf-8") as f:
                nodes_data = json.load(f)
            embeddings = np.load(str(emb_path))
        except Exception as e:
            print(f"⚠ Failed to load nodes or embeddings: {e}")
            return []

        nodes: List[TextNode] = []
        for i, item in enumerate(nodes_data):
            text = item.get("text", "")
            metadata = item.get("metadata", {})
            node_id = item.get("id", f"node-{i}")
            node = TextNode(text=text, metadata=metadata, id_=node_id)
            if i < len(embeddings):
                node.embedding = embeddings[i].tolist()
            nodes.append(node)

        return nodes
    
    def _mean_pooling(self, output, mask):
        """Attention-aware mean pooling"""
        embeddings = output[0]
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    
    @torch.no_grad()
    def embed_query(self, query: str) -> np.ndarray:
        """Generate query embedding - OPTIMIZED"""
        try:
            inputs = self.tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
            
            try:
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            except Exception:
                pass
            
            output = self.model(**inputs)
            embeddings = self._mean_pooling(output, inputs['attention_mask'])
            return embeddings.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error embedding query: {e}")
            return np.zeros(768)
    
    def retrieve_with_quality_filtering(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        patient_context: Optional[Dict] = None
    ) -> Tuple[List[Dict], Dict]:
        """FAST retrieval with quality filtering + Knowledge Graph reasoning"""
        if self.embeddings_matrix.size == 0:
            return [], {"error": "No embeddings available"}
            
        top_k = top_k or self.config.top_k_retrieval
        
        # Step 1: Vector retrieval
        query_embedding = self.embed_query(query).reshape(1, -1)
        all_sims = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
        top_indices = np.argsort(all_sims)[::-1][:top_k * 2]  # Get more candidates for graph boosting
        
        results = []
        for idx in top_indices:
            score = float(all_sims[idx])
            if score >= self.config.min_relevance_score:
                node = self.nodes[idx]
                clean_title = self._clean_title(node.metadata.get('document_title', 'Unknown'), node.text)
                
                results.append({
                    'text': node.text,
                    'score': score,
                    'node_id': node.id_,
                    'index': int(idx),
                    'document_name': node.metadata.get('document_name', 'Unknown'),
                    'document_title': clean_title,
                    'page_number': node.metadata.get('page_number', 0),
                    'content_type': node.metadata.get('content_type', 'text'),
                    'relevance_score': node.metadata.get('relevance_score', 0.0)
                })
        
        # Step 2: Knowledge graph reasoning (if patient context provided)
        graph_insights = {}
        if patient_context:
            try:
                graph_insights = self.kg.reason_about_patient(patient_context)
                
                # Boost documents that mention indicated stages
                if graph_insights.get('stage_indicators'):
                    indicated_stages = [ind['indicates'] for ind in graph_insights['stage_indicators']]
                    
                    for result in results:
                        text_lower = result['text'].lower()
                        for stage in indicated_stages:
                            if stage.lower() in text_lower:
                                result['score'] *= 1.2  # 20% boost for graph-relevant docs
                                result['graph_boosted'] = True
                    
                    # Re-sort by boosted scores
                    results.sort(key=lambda x: x['score'], reverse=True)
            except Exception as e:
                print(f"⚠ Graph reasoning error: {e}")
        
        metrics = {
            'total_retrieved': len(results),
            'avg_relevance': np.mean([r['score'] for r in results]) if results else 0.0,
            'max_relevance': max([r['score'] for r in results]) if results else 0.0,
            'min_relevance': min([r['score'] for r in results]) if results else 0.0,
            'graph_insights': graph_insights,
            'graph_enhanced': bool(patient_context and graph_insights)
        }
        
        return results[:self.config.rerank_top_k], metrics
    
    def _clean_title(self, raw_title: str, text: str) -> str:
        """Extract clean, meaningful title"""
        if not raw_title or raw_title.lower() in ['unknown', 'n/a']:
            lines = text.split('\n')
            for line in lines[:5]:
                line = line.strip()
                if (20 < len(line) < 150 and 
                    ' ' in line and 
                    not re.search(r'(doi|issn|copyright|published|received|accepted):', line, re.I)):
                    return line
            return "Medical Research Article"
        
        title = raw_title
        title = re.sub(r'^\d+\s*', '', title)
        title = re.sub(r'(doi|issn|citation)[:.\s].*', '', title, flags=re.I)
        title = re.sub(r'copyright.*', '', title, flags=re.I)
        title = re.sub(r'published.*', '', title, flags=re.I)
        title = re.sub(r'received:.*', '', title, flags=re.I)
        title = re.sub(r'accepted:.*', '', title, flags=re.I)
        title = re.sub(r'\s+', ' ', title).strip()
        
        if len(title) < 15:
            return "Alzheimer's Disease Research"
        
        return title[:120]


class QueryAnalyzer:
    """Fast query classification"""
    
    QUERY_PATTERNS = {
        QueryType.SYMPTOM_INQUIRY: ['symptom', 'sign', 'early', 'memory', 'forget', 'confusion'],
        QueryType.TREATMENT_OPTIONS: ['treatment', 'medication', 'therapy', 'drug', 'medicine'],
        QueryType.LIFESTYLE_ADVICE: ['diet', 'exercise', 'lifestyle', 'prevention', 'nutrition'],
        QueryType.CAREGIVER_SUPPORT: ['caregiver', 'caring', 'support', 'help', 'family'],
        QueryType.DIAGNOSIS_INFO: ['diagnosis', 'test', 'assessment', 'screening', 'biomarker'],
        QueryType.RISK_FACTORS: ['risk', 'factor', 'prevent', 'cause'],
        QueryType.GREETING: ['hi', 'hello', 'hey', 'good morning', 'good evening', 'bonjour', 'greetings'],
        QueryType.GRATITUDE: ['thank you', 'thanks', 'thank', 'merci', 'appreciate', 'grateful']
    }

    @classmethod
    def classify_query(cls, query: str) -> Tuple[QueryType, str]:
        """Classify query and return explanation for transparency"""
        q = query.strip().lower()
        
        # Explicit short greeting/gratitude detection
        if len(q.split()) <= 4:
            for g in cls.QUERY_PATTERNS[QueryType.GREETING]:
                if g == q or q.startswith(g):
                    return QueryType.GREETING, f"Detected greeting pattern: '{g}'"
            
            for t in cls.QUERY_PATTERNS[QueryType.GRATITUDE]:
                if t in q:
                    return QueryType.GRATITUDE, f"Detected gratitude pattern: '{t}'"
        
        # Normal scoring
        scores = {}
        for qtype, keywords in cls.QUERY_PATTERNS.items():
            if qtype in [QueryType.GREETING, QueryType.GRATITUDE]:
                continue
            scores[qtype] = sum(2 if kw in q else 0 for kw in keywords)
        
        max_score = max(scores.values()) if scores else 0
        if max_score > 0:
            best_type = max(scores, key=scores.get)
            matched_keywords = [kw for kw in cls.QUERY_PATTERNS[best_type] if kw in q]
            return best_type, f"Matched keywords: {', '.join(matched_keywords)}"
        
        return QueryType.GENERAL, "No specific category matched"


class AgenticResponseGenerator:
    """OPTIMIZED response generator with formatting"""
    
    def __init__(self, retriever: EnhancedRetriever, config: RAGConfig):
        self.retriever = retriever
        self.config = config
        self.groq_client = client
    
    def _build_optimized_system_prompt(self, user_type: UserType) -> str:
        """Distinct persona prompts with direct answer constraints"""
        if user_type == UserType.PATIENT:
            return """You are a warm, caring AI companion helping people and families understand memory concerns and Alzheimer's disease.

**Communication Style:**
- Use warm, encouraging, and conversational language
- Explain medical concepts in simple, everyday terms
- Be empathetic and supportive while remaining accurate
- Provide reassurance and hope while being realistic

**Answer Directly:**
- Answer the specific question asked in 2-3 clear paragraphs
- Do NOT ask follow-up questions unless the user's question is unclear
- Do NOT suggest "you might also want to know..." or similar expansions
- Focus only on what was asked

**Evidence:**
- Integrate research naturally: "Research shows..." or "Studies have found..."
- Recommend consulting healthcare professionals when appropriate

**Key Constraint:** Answer directly. No unsolicited advice or follow-up questions."""
        elif user_type == UserType.DOCTOR:
            return """You are an expert medical AI assistant specializing in Alzheimer's disease and neurodegenerative disorders.

**Communication Style:**
- Use precise medical vocabulary and technical terminology
- Explain pathophysiology, mechanisms, and complex concepts in depth
- Be professional, evidence-based, and clinically accurate
- Maintain natural flow—avoid being robotic

**Answer Directly:**
- Provide the core answer in 2-3 well-structured paragraphs
- Do NOT ask follow-up questions
- Do NOT suggest additional topics unless explicitly requested
- Focus on the specific clinical/scientific question asked

**Evidence:**
- Cite sources when relevant
- Reference specific studies, trials, or guidelines when applicable
- Use technical precision appropriate for medical professionals

**Key Constraint:** Answer directly and comprehensively. No unsolicited suggestions."""
        else:  # CAREGIVER
            return """You are a supportive AI assistant for caregivers of people with Alzheimer's and dementia.

**Communication Style:**
- Be warm, practical, and empathetic
- Provide actionable guidance and emotional support
- Balance medical accuracy with accessible language

**Answer Directly:**
- Answer the specific caregiving question in 2-3 paragraphs
- Do NOT ask follow-up questions
- Focus on practical, helpful information

**Key Constraint:** Answer directly. No unsolicited advice."""
     
    def generate_greeting(
        self,
        patient_context: Optional[PatientContext],
        user_type: UserType
    ) -> ExplainableResult:
        """Minimal, personalized greeting (no retrieval, no LLM)."""
        # Build a human, concise context line
        ctx_phrases = []
        if patient_context:
            if patient_context.symptoms:
                listed = ", ".join(patient_context.symptoms[:2])
                more = len(patient_context.symptoms) - 2
                if more > 0:
                    listed += f" (+{more} more)"
                ctx_phrases.append(f"you reported {listed}")
            if patient_context.age is not None:
                ctx_phrases.append(f"you're {patient_context.age}")
            if patient_context.mmse_score is not None:
                ctx_phrases.append(f"MMSE {patient_context.mmse_score}/30")
            if patient_context.moca_score is not None:
                ctx_phrases.append(f"MoCA {patient_context.moca_score}/30")

        context_line = ""
        if ctx_phrases:
            # Prefer symptoms first, then age/tests for a natural feel
            context_line = " I noticed " + "; ".join(ctx_phrases) + "."

        # Tone by user type
        if user_type == UserType.CAREGIVER:
            role_line = "I can help you plan support and answer caregiving questions."
        elif user_type == UserType.DOCTOR:
            role_line = "I can assist with concise, evidence‑based clinical queries."
        else:
            role_line = "I can help you make sense of memory and thinking changes."

        # Offer a clear next step without sounding generic
        suggestions = "If you’d like, ask about what these signs might mean, which tests are useful now, or practical steps to reduce risk."

        answer = f"Hi.{context_line} {role_line} {suggestions}"

        reasoning = [ReasoningStep(
            step_number=1,
            action="Greeting",
            reasoning="Detected greeting; generated short, contextual welcome without retrieval.",
            result="Short personalized welcome",
            confidence=0.95,
            timestamp=datetime.now().isoformat()
        )]

        return ExplainableResult(
            answer=answer.strip(),
            reasoning_trace=reasoning,
            confidence_score=0.95,
            sources=[],
            retrieval_metrics={},
            generation_metrics={"mode": "greeting"},
            warnings=[],
            recommendations=["Ask a focused question about symptoms, tests, or next steps"],
            patient_summary=patient_context.generate_patient_summary() if patient_context else None
        )
    

    def generate_gratitude(
        self,
        patient_context: Optional[PatientContext],
        user_type: UserType
    ) -> ExplainableResult:
        """Brief, polite acknowledgment of gratitude (no retrieval, no LLM)."""
        
        # Simple, human responses
        if user_type == UserType.DOCTOR:
            answer = "You're welcome. Feel free to ask if you need more information."
        elif user_type == UserType.CAREGIVER:
            answer = "You're very welcome. Caregiving is challenging work—I'm glad I could help."
        else:  # PATIENT
            answer = "You're welcome. I'm here whenever you have questions or concerns."
        
        reasoning = [ReasoningStep(
            step_number=1,
            action="Gratitude Response",
            reasoning="Detected gratitude expression; generated brief acknowledgment.",
            result="Polite acknowledgment",
            confidence=0.98,
            timestamp=datetime.now().isoformat()
        )]
        
        return ExplainableResult(
            answer=answer,
            reasoning_trace=reasoning,
            confidence_score=0.98,
            sources=[],
            retrieval_metrics={},
            generation_metrics={"mode": "gratitude"},
            warnings=[],
            recommendations=[],
            patient_summary=patient_context.generate_patient_summary() if patient_context else None
        )
    
    def generate_with_reasoning(
        self, 
        query: str, 
        context_docs: List[Dict], 
        query_type: QueryType,
        user_type: UserType,
        patient_context: Optional[PatientContext] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> ExplainableResult:
        """Generate response with reasoning - OPTIMIZED"""
        
        reasoning_trace = []
        warnings = []
        recommendations = []
        
        # Generate patient summary if context provided
        patient_summary = None
        if patient_context:
            patient_summary = patient_context.generate_patient_summary()
        
        reasoning_trace.append(ReasoningStep(
            step_number=1,
            action="Query Analysis",
            reasoning=f"Classified as {query_type.value} query",
            result=f"Type: {query_type.value}",
            confidence=0.9,
            timestamp=datetime.now().isoformat()
        ))
        
        if not context_docs:
            warnings.append("No relevant sources found")
            return self._create_no_evidence_response(query, reasoning_trace, warnings, patient_summary)
        
        avg_relevance = np.mean([d['score'] for d in context_docs])
        reasoning_trace.append(ReasoningStep(
            step_number=2,
            action="Evidence Assessment",
            reasoning=f"Retrieved {len(context_docs)} high-quality sources",
            result=f"Avg relevance: {avg_relevance:.0%}",
            confidence=float(avg_relevance),
            timestamp=datetime.now().isoformat()
        ))
        
        answer, generation_metadata = self._generate_response_optimized(
            query, context_docs, query_type, user_type, patient_context, history
        )
        
        confidence = self._calculate_confidence(answer, context_docs, generation_metadata)
        
        reasoning_trace.append(ReasoningStep(
            step_number=3,
            action="Response Generation",
            reasoning=f"Generated patient-friendly response",
            result="Complete",
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        ))
        
        if confidence < self.config.min_confidence_threshold:
            warnings.append(f"Moderate confidence. Please consult healthcare professional.")
        
        if user_type == UserType.PATIENT:
            recommendations.append("Discuss with your doctor for personalized advice")
        
        sources = self._build_clean_sources(context_docs)
        
        return ExplainableResult(
            answer=answer,
            reasoning_trace=reasoning_trace,
            confidence_score=confidence,
            sources=sources,
            retrieval_metrics={'avg_relevance': avg_relevance},
            generation_metrics=generation_metadata,
            warnings=warnings,
            recommendations=recommendations,
            patient_summary=patient_summary
        )
    
    def _generate_response_optimized(
        self, 
        query: str, 
        context_docs: List[Dict], 
        query_type: QueryType,
        user_type: UserType,
        patient_context: Optional[PatientContext],
        history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[str, Dict]:
        """FAST optimized response generation with STRUCTURED output"""
        
        context_parts = []
        for i, doc in enumerate(context_docs[:4], 1):
            text = doc.get('text', '')
            text = re.sub(r'\[\d+\]', '', text)
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'(doi|issn|copyright).*', '', text, flags=re.I)
            text = re.sub(r'\s+', ' ', text).strip()
            excerpt = text[:400]
            title = doc.get('document_title', 'Medical Research')
            context_parts.append(f"[Source {i}] {title}\n{excerpt}\n")
        
        patient_info = ""
        if patient_context:
            ctx_str = patient_context.to_context_string()
            if ctx_str:
                patient_info = f"\nPatient: {ctx_str}\n"

        history_str = ""
        if history:
            history_str = "\nConversation History:\n"
            for msg in history[-5:]:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                history_str += f"{role.capitalize()}: {content}\n"
        
        system_prompt = self._build_optimized_system_prompt(user_type)
        
        user_prompt = f"""{chr(10).join(context_parts)}
{patient_info}
{history_str}
Question: {query}

IMPORTANT: Structure your response using the format specified in the system prompt. Use **bold** for key terms, clear section headers, and numbered/bulleted lists. Make it easy to read and scan.

Provide a clear, helpful answer in 3-4 well-structured paragraphs with formatting."""

        try:
            response = self.groq_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.patient_temperature if user_type == UserType.PATIENT else self.config.doctor_temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                timeout=15
            )
            
            answer = response.choices[0].message.content if response.choices else "No response"
            answer = self._clean_answer(answer)
            
            metadata = {
                'model': self.config.llm_model,
                'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') else 0,
                'completion_tokens': getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') else 0
            }
            
            # Build ExplainableResult consistently
            confidence = self._calculate_confidence(answer, context_docs, metadata)
            
            # Only include sources actually cited in the answer
            cited_docs = self._extract_cited_sources(answer, context_docs or [])
            sources = [
                {
                    "title": d.get("document_title") or d.get("document_name") or "Unknown",
                    "page": d.get("page_number") or d.get("page") or "N/A",
                    "excerpt": (d.get("text") or "")[:300],
                    "relevance": float(d.get("score", 0.0))
                } for d in cited_docs
            ]
            
            retrieval_metrics = {
                "total_retrieved": len(context_docs or []),
                "total_cited": len(cited_docs),
                "avg_relevance": float(np.mean([d['score'] for d in (context_docs or [])])) if context_docs else 0.0
            }
            
            return answer, metadata
            
        except Exception as e:
            print(f"Generation error: {e}")
            return self._create_fallback_answer(query, context_docs), {'error': str(e)}
    
    def _clean_answer(self, answer: str) -> str:
        """Clean the AI answer - remove excessive formatting"""
        if not answer:
            return "I apologize, but I couldn't generate a response."
        
        # Ensure newlines before headers and horizontal rules
        answer = re.sub(r'(?<!\n)###', '\n\n###', answer)
        answer = re.sub(r'(?<!\n)---', '\n\n---\n\n', answer)
        answer = re.sub(r'(?<!\n)\*\*', '\n**', answer) # Optional: bold on new line if it looks like a list item
        
        # Remove excessive stars/hyphens but keep markdown formatting
        answer = re.sub(r'\*{3,}', '**', answer)  # Triple+ stars to double
        answer = re.sub(r'-{4,}', '', answer)  # Remove long hyphen lines
        answer = re.sub(r'_{3,}', '', answer)  # Remove long underscores
        
        # Clean whitespace
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        answer = re.sub(r' {2,}', ' ', answer)
        
        # Remove metadata
        answer = re.sub(r'(doi|issn):.*?\n', '', answer, flags=re.I)
        
        return answer.strip()
    
    def _build_clean_sources(self, context_docs: List[Dict]) -> List[Dict]:
        """Build clean, meaningful sources"""
        sources = []
        for i, doc in enumerate(context_docs, 1):
            title = doc.get('document_title', f"Medical Research {i}")
            text = doc.get('text', '')
            text = re.sub(r'\[\d+\]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            excerpt = text[:200]
            if len(text) > 200:
                excerpt = excerpt.rsplit(' ', 1)[0] + '...'
            
            sources.append({
                'source_id': i,
                'title': title,
                'document_title': title,
                'page': doc.get('page_number', 'N/A'),
                'page_number': doc.get('page_number', 'N/A'),
                'relevance': round(doc.get('score', 0) * 100, 1),
                'score': doc.get('score', 0),
                'type': doc.get('content_type', 'text'),
                'excerpt': excerpt,
                'text': excerpt
            })
        
        return sources
    
    def _calculate_confidence(
        self, 
        answer: str, 
        context_docs: List[Dict],
        generation_metadata: Dict
    ) -> float:
        """Calculate confidence score"""
        
        if not context_docs:
            return 0.0
        
        # Factor 1: Retrieval quality (40%)
        avg_relevance = np.mean([d['score'] for d in context_docs])
        retrieval_score = avg_relevance * 0.4
        
        # Factor 2: Answer length and structure (30%)
        answer_length = len(answer.split())
        length_score = min(answer_length / 200, 1.0) * 0.3
        
        # Factor 3: Source citations (30%)
        citation_count = answer.count('[Source')
        citation_score = min(citation_count / len(context_docs), 1.0) * 0.3
        
        confidence = retrieval_score + length_score + citation_score
        
        return min(confidence, 1.0)

    def _extract_cited_sources(self, answer: str, context_docs: List[Dict]) -> List[Dict]:
        """Extract only sources that were actually cited in the answer"""
        cited_sources = []
        
        # Find all [Source N] citations in answer
        citation_pattern = r'\[Source[s]?\s+(\d+)\]'
        cited_numbers = set()
        for match in re.finditer(citation_pattern, answer, re.IGNORECASE):
            try:
                cited_numbers.add(int(match.group(1)))
            except:
                pass
        
        # If citations found, return only cited sources
        if cited_numbers:
            for i, doc in enumerate(context_docs, 1):
                if i in cited_numbers:
                    cited_sources.append(doc)
        else:
            # Fallback: return top 3 most relevant if no explicit citations
            cited_sources = sorted(context_docs, key=lambda x: x.get('score', 0), reverse=True)[:3]
        
        return cited_sources
    
    def _create_fallback_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Quick fallback when AI fails"""
        if not context_docs:
            return f"""I couldn't find specific information about: "{query}"

Please try:
• Rephrasing your question
• Asking about a specific aspect
• Consulting with an Alzheimer's specialist

Remember: Professional medical advice is essential for health concerns."""
        
        text = context_docs[0].get('text', '')[:300]
        text = re.sub(r'\s+', ' ', text).strip()
        
        return f"""Based on medical literature about "{query}":

{text}

For personalized guidance, please consult with a healthcare professional."""
    
    def _create_no_evidence_response(
        self, 
        query: str, 
        reasoning_trace: List[ReasoningStep],
        warnings: List[str],
        patient_summary: Optional[str] = None
    ) -> ExplainableResult:
        """Response when no evidence found"""
        
        answer = f"""I couldn't find specific information about: "{query}"

This might mean:
• Your question needs expert medical consultation
• The information may not be in my current database
• It requires personalized medical assessment

**Recommendation:** Consult with a neurologist or Alzheimer's specialist for personalized guidance."""

        return ExplainableResult(
            answer=answer,
            reasoning_trace=reasoning_trace,
            confidence_score=0.0,
            sources=[],
            retrieval_metrics={'total_retrieved': 0},
            generation_metrics={},
            warnings=warnings,
            recommendations=["Consult healthcare professional"],
            patient_summary=patient_summary
        )


class RAGResult(BaseModel):
    answer: str
    confidence: float
    key_findings: List[str] = []
    recommendations: List[str] = []
    citations: List[Dict[str, str]] = []  # [{title,url,snippet}]

class AlzheimerRAGSystem:
    """OPTIMIZED agentic RAG system"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        
        print("\n" + "="*70)
        print("ENHANCED AGENTIC RAG SYSTEM v3.2 - OPTIMIZED WITH FORMATTING")
        print("="*70)
        
        self.retriever = EnhancedRetriever(self.config)
        self.generator = AgenticResponseGenerator(self.retriever, self.config)
        self.analyzer = QueryAnalyzer()
        
        print("System initialized successfully!\n")
    
    def query(
        self,
        user_query: str,
        user_type: UserType = UserType.PATIENT,
        patient_context: Optional[PatientContext] = None,
        explain: bool = True,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict:
        print(f"\n{'='*70}")
        print(f"🔍 QUERY: {user_query[:80]}...")
        print(f"{'='*70}\n")

        query_type, category_explanation = self.analyzer.classify_query(user_query)
        print(f"🏷️  Category: {query_type.value}")
        print(f"📝 Reason: {category_explanation}")

        # Fast path for greeting: no retrieval / LLM
        if query_type == QueryType.GREETING:
            print("👋 Greeting detected: skipping retrieval and generation.")
            result = self.generator.generate_greeting(patient_context, user_type)
            cleaned_answer = re.sub(r'[\r\n]+', '\n', result.answer).strip()
            final = {
                'answer': cleaned_answer,
                'short_summary': cleaned_answer,
                'patient_summary': result.patient_summary,
                'confidence': result.confidence_score,
                'query_type': query_type.value,
                'category_explanation': category_explanation,
                'user_type': user_type.value,
                'sources': [],
                'reasoning_trace': [
                    {
                        'step': s.step_number,
                        'action': s.action,
                        'reasoning': s.reasoning,
                        'confidence': s.confidence
                    } for s in result.reasoning_trace
                ] if explain else [],
                'retrieval_metrics': {},
                'warnings': result.warnings,
                'recommendations': result.recommendations,
                'patient_context_used': patient_context is not None
            }
            return final

        # Fast path for gratitude: no retrieval / LLM
        if query_type == QueryType.GRATITUDE:
            print("🙏 Gratitude detected: generating acknowledgment.")
            result = self.generator.generate_gratitude(patient_context, user_type)
            cleaned_answer = re.sub(r'[\r\n]+', '\n', result.answer).strip()
            final = {
                'answer': cleaned_answer,
                'short_summary': cleaned_answer,
                'patient_summary': result.patient_summary,
                'confidence': result.confidence_score,
                'query_type': query_type.value,
                'category_explanation': category_explanation,
                'user_type': user_type.value,
                'sources': [],
                'reasoning_trace': [
                    {
                        'step': s.step_number,
                        'action': s.action,
                        'reasoning': s.reasoning,
                        'confidence': s.confidence
                    } for s in result.reasoning_trace
                ] if explain else [],
                'retrieval_metrics': {},
                'warnings': result.warnings,
                'recommendations': result.recommendations,
                'patient_context_used': patient_context is not None
            }
            return final

        print("📚 Retrieving...")
        # Convert PatientContext to dict for knowledge graph
        patient_dict = None
        if patient_context:
            patient_dict = {
                'age': patient_context.age,
                'mmse_score': patient_context.mmse_score,
                'moca_score': patient_context.moca_score,
                'biomarkers': patient_context.biomarkers,
                'symptoms': patient_context.symptoms
            }
        retrieved_docs, retrieval_metrics = self.retriever.retrieve_with_quality_filtering(user_query, patient_context=patient_dict)
        print(f"✓ Retrieved {len(retrieved_docs)} sources ({retrieval_metrics.get('avg_relevance', 0):.0%} avg relevance)")
        print("🤖 Generating...")
        result = self.generator.generate_with_reasoning(
            user_query,
            retrieved_docs,
            query_type,
            user_type,
            patient_context,
            history
        )
        print(f"✓ Complete (Confidence: {result.confidence_score:.0%})\n")

        raw_answer = result.answer or ""
        cleaned_answer = re.sub(r'[\r\n]+', '\n', raw_answer)
        cleaned_answer = re.sub(r'\s{2,}', ' ', cleaned_answer).strip()
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned_answer) if s.strip()]
        # Build concise reasoning-only summary (doctor-facing) – strip recommendations/next steps
        short_summary = ""
        if sentences:
            core_sents = sentences[:2]
            core_sents = [
                re.sub(r"(?i)\b(recommend|consider|should|advise|suggest|next step|plan)\b.*", "", s).strip()
                for s in core_sents
            ]
            core_sents = [s for s in core_sents if s]
            short_summary = " ".join(core_sents).strip() or (cleaned_answer[:250] + "..." if len(cleaned_answer) > 250 else cleaned_answer)
        else:
            short_summary = cleaned_answer[:250] + "..." if len(cleaned_answer) > 250 else cleaned_answer
        # Append compact source hint if available
        if result.sources:
            src_titles = [(src.get('title') or 'Source')[:60] for src in result.sources[:2]]
            if any(src_titles):
                short_summary = (short_summary + (" " if short_summary else "")) + f"[Sources: {'; '.join([t for t in src_titles if t])}]"
        if short_summary and not re.search(r'[.!?]$', short_summary):
            short_summary += '.'

        final_result = {
            'answer': cleaned_answer,
            'short_summary': short_summary,
            'patient_summary': result.patient_summary,
            'confidence': result.confidence_score,
            'query_type': query_type.value,
            'category_explanation': category_explanation,
            'user_type': user_type.value,
            'sources': result.sources if result.sources else [],
            'reasoning_trace': [
                {
                    'step': step.step_number,
                    'action': step.action,
                    'reasoning': step.reasoning,
                    'confidence': step.confidence
                }
                for step in result.reasoning_trace
            ] if explain else [],
            'retrieval_metrics': retrieval_metrics,
            'warnings': result.warnings,
            'recommendations': result.recommendations,
            'patient_context_used': patient_context is not None
        }
        return final_result

    def run_query(self, query: str, patient_profile: Dict[str, Any], history: List[Dict[str, Any]] = None) -> RAGResult:
        # retrieve
        docs = self.retriever.search(query, top_k=5, metadata=patient_profile)
        # Build citations from retrieved docs
        citations: List[Dict[str, str]] = []
        for d in docs:
            citations.append({
                "title": (getattr(d, "metadata", {}) or {}).get("title") or "Unknown",
                "url": (getattr(d, "metadata", {}) or {}).get("url") or "",
                "snippet": (getattr(d, "content", "")[:240] + "…") if getattr(d, "content", "") else ""
            })
        # Minimal placeholder answer to keep system stable
        answer_text = "I'll review the most relevant sources and provide a concise, patient-specific summary."
        return RAGResult(
            answer=answer_text,
            confidence=0.0,
            key_findings=[],
            recommendations=[],
            citations=citations
        )