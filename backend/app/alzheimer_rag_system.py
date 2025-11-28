import os
import pickle
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from groq import Groq
from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer, AutoModel
from llama_index.core.schema import TextNode

# use package import to avoid local circular import issues
from config import PDF_DIR, OUT_DIR, CHUNK_DIR, INDEX_PATH, NODES_PATH, EMBEDDING_MODEL_NAME, DEVICE, GEN_MODEL_NAME


@dataclass
class RAGConfig:
    """Centralized configuration for the RAG system"""
    # Paths
    pdf_dir: str = PDF_DIR
    output_dir: str = OUT_DIR
    embeddings_path: str = NODES_PATH
    index_path: str = INDEX_PATH

    # Model settings
    embedding_model: str = EMBEDDING_MODEL_NAME
    llm_model: str = GEN_MODEL_NAME

    # Chunking
    chunk_size: int = 3
    overlap_size: int = 1

    # Retrieval
    top_k_retrieval: int = 5
    confidence_threshold: float = 0.5

    # Generation
    temperature: float = 0.3
    max_tokens: int = 2000

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"



class UserType(Enum):
    """User types for personalized responses"""
    DOCTOR = "doctor"
    PATIENT = "patient"
    CAREGIVER = "caregiver"


class QueryType(Enum):
    
    SYMPTOM_INQUIRY = "symptom"
    TREATMENT_OPTIONS = "treatment"
    LIFESTYLE_ADVICE = "lifestyle"
    CAREGIVER_SUPPORT = "caregiver"
    DIAGNOSIS_INFO = "diagnosis"
    RESEARCH_UPDATE = "research"
    MRI_INTERPRETATION = "mri"
    GENERAL = "general"


class AlzheimerStage(Enum):
    """Disease stages"""
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
    mri_result: Optional[Dict] = None  # Will contain CNN model predictions
    stage: AlzheimerStage = AlzheimerStage.UNKNOWN
    medical_history: List[str] = None
    current_medications: List[str] = None
    
    def __post_init__(self):
        if self.symptoms is None:
            self.symptoms = []
        if self.medical_history is None:
            self.medical_history = []
        if self.current_medications is None:
            self.current_medications = []
    
    def to_context_string(self) -> str:
        """Convert patient context to string for LLM"""
        parts = []
        if self.age:
            parts.append(f"Age: {self.age}")
        if self.symptoms:
            parts.append(f"Symptoms: {', '.join(self.symptoms)}")
        if self.mri_result:
            confidence = self.mri_result.get('confidence', 0)
            prediction = self.mri_result.get('prediction', 'unknown')
            parts.append(f"MRI Analysis: {prediction} (confidence: {confidence:.2%})")
        if self.stage != AlzheimerStage.UNKNOWN:
            parts.append(f"Disease Stage: {self.stage.value}")
        if self.current_medications:
            parts.append(f"Current Medications: {', '.join(self.current_medications)}")
        return "\n".join(parts) if parts else "No patient context available"



class MRIAnalyzer:
    """
    Placeholder for MRI CNN model integration
    Your friend will implement the actual CNN model
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        # TODO: Load CNN model when available
    
    def analyze_mri(self, mri_image_path: str) -> Dict:
        """
        Analyze MRI image and return prediction
        
        Returns:
            Dict with 'prediction', 'confidence', 'stage'
        """
        # Placeholder implementation
        # Your friend will replace this with actual CNN inference
        return {
            'prediction': 'alzheimer_detected',
            'confidence': 0.85,
            'stage': AlzheimerStage.MILD_IMPAIRMENT.value,
            'explanation': 'Placeholder: Hippocampal atrophy detected in temporal lobe',
            'model_used': 'CNN_XAI_Model_v1'
        }


# ============================================================================
# EMBEDDING & RETRIEVAL
# ============================================================================

class PubMedBERTRetriever:
    """Handles embedding and retrieval using PubMedBERT"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        print(f"🔧 Initializing PubMedBERT Retriever on {config.device}...")
        
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
        self.model = AutoModel.from_pretrained(config.embedding_model)
        self.model.eval()
        self.model.to(config.device)
        
        # Load pre-computed nodes
        self.nodes = self._load_nodes()
        self.embeddings_matrix = np.array([node.embedding for node in self.nodes])
        
        print(f"✓ Loaded {len(self.nodes)} document chunks")
    
    def _load_nodes(self) -> List[TextNode]:
        """Load pre-computed embeddings"""
        if not Path(self.config.embeddings_path).exists():
            raise FileNotFoundError(
                f"Embeddings not found at {self.config.embeddings_path}. "
                "Please run document processing first."
            )
        with open(self.config.embeddings_path, "rb") as f:
            return pickle.load(f)
    
    def _mean_pooling(self, output, mask):
        """Attention-aware mean pooling"""
        embeddings = output[0]
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query"""
        inputs = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.config.device)
        
        with torch.no_grad():
            output = self.model(**inputs)
        
        embeddings = self._mean_pooling(output, inputs['attention_mask'])
        return embeddings.cpu().numpy().flatten()
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve most relevant documents"""
        top_k = top_k or self.config.top_k_retrieval
        
        query_embedding = self.embed_query(query).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.nodes[idx].text,
                'score': float(similarities[idx]),
                'node_id': self.nodes[idx].id_,
                'index': int(idx)
            })
        
        return results


# ============================================================================
# QUERY ANALYSIS
# ============================================================================

class QueryAnalyzer:
    """Analyzes and classifies user queries"""
    
    QUERY_PATTERNS = {
        QueryType.SYMPTOM_INQUIRY: [
            'symptom', 'sign', 'indicator', 'early stage', 
            'warning', 'behavior change', 'memory loss', 'forget'
        ],
        QueryType.TREATMENT_OPTIONS: [
            'treatment', 'medication', 'therapy', 'drug',
            'cure', 'intervention', 'clinical trial', 'medicine'
        ],
        QueryType.LIFESTYLE_ADVICE: [
            'diet', 'exercise', 'lifestyle', 'prevention',
            'nutrition', 'activity', 'sleep', 'routine', 'food'
        ],
        QueryType.CAREGIVER_SUPPORT: [
            'caregiver', 'caring for', 'support', 'help',
            'family', 'burden', 'stress', 'managing', 'cope'
        ],
        QueryType.DIAGNOSIS_INFO: [
            'diagnosis', 'test', 'assessment', 'screening',
            'detect', 'identify', 'biomarker', 'diagnose'
        ],
        QueryType.MRI_INTERPRETATION: [
            'mri', 'scan', 'imaging', 'brain scan', 'ct scan',
            'pet scan', 'neuroimaging'
        ],
        QueryType.RESEARCH_UPDATE: [
            'research', 'study', 'latest', 'new',
            'breakthrough', 'clinical', 'trial', 'findings'
        ]
    }
    
    @classmethod
    def classify_query(cls, query: str) -> QueryType:
        """Classify query type based on keywords"""
        query_lower = query.lower()
        
        scores = {}
        for qtype, keywords in cls.QUERY_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[qtype] = score
        
        max_score = max(scores.values())
        if max_score == 0:
            return QueryType.GENERAL
        
        return max(scores, key=scores.get)
    
    @classmethod
    def extract_entities(cls, query: str) -> Dict[str, List[str]]:
        """Extract medical entities from query"""
        query_lower = query.lower()
        
        entities = {
            'medications': [],
            'symptoms': [],
            'activities': [],
            'tests': []
        }
        
        # Medication keywords
        meds = ['donepezil', 'memantine', 'rivastigmine', 'galantamine', 
                'aducanumab', 'lecanemab']
        # Symptom keywords
        symptoms = ['memory', 'confusion', 'disorientation', 'aggression', 
                   'wandering', 'agitation', 'depression']
        # Activity keywords
        activities = ['exercise', 'walk', 'cognitive', 'social', 'diet', 
                     'meditation', 'reading']
        # Test keywords
        tests = ['mri', 'ct', 'pet', 'spinal tap', 'blood test', 'cognitive test']
        
        for med in meds:
            if med in query_lower:
                entities['medications'].append(med)
        for sym in symptoms:
            if sym in query_lower:
                entities['symptoms'].append(sym)
        for act in activities:
            if act in query_lower:
                entities['activities'].append(act)
        for test in tests:
            if test in query_lower:
                entities['tests'].append(test)
        
        return entities


# ============================================================================
# XAI EXPLAINER
# ============================================================================

class XAIExplainer:
    """Provides explainability for RAG decisions"""
    
    @staticmethod
    def explain_retrieval(retrieved_docs: List[Dict], query: str) -> Dict:
        """Explain document retrieval decisions"""
        if not retrieved_docs:
            return {
                'confidence': 0.0,
                'reasoning': ['⚠ No relevant documents found'],
                'suggestion': 'Try rephrasing your query or use the MCP system to search online resources'
            }
        
        scores = [doc['score'] for doc in retrieved_docs]
        avg_score = np.mean(scores)
        variance = np.var(scores)
        
        explanation = {
            'confidence': avg_score,
            'top_score': scores[0],
            'score_distribution': scores,
            'reasoning': [],
            'suggestion': None
        }
        
        # Confidence assessment
        if avg_score > 0.7:
            explanation['reasoning'].append(
                "✓ High confidence: Retrieved documents are highly relevant"
            )
        elif avg_score > 0.5:
            explanation['reasoning'].append(
                "⚠ Moderate confidence: Documents are somewhat relevant"
            )
        else:
            explanation['reasoning'].append(
                "⚠ Low confidence: Limited relevant information in local documents"
            )
            explanation['suggestion'] = "Consider triggering MCP workflow for external search"
        
        # Variance assessment
        if variance < 0.01:
            explanation['reasoning'].append(
                "✓ Consistent relevance across all documents"
            )
        else:
            explanation['reasoning'].append(
                "⚠ Varied relevance: Top documents significantly more relevant"
            )
        
        return explanation
    
    @staticmethod
    def explain_mri_integration(mri_result: Dict) -> str:
        """Explain MRI analysis results"""
        if not mri_result:
            return "No MRI analysis available"
        
        confidence = mri_result.get('confidence', 0)
        prediction = mri_result.get('prediction', 'unknown')
        explanation = mri_result.get('explanation', 'No details provided')
        
        return f"""
🧠 **MRI Analysis Explanation:**
- Prediction: {prediction}
- Confidence: {confidence:.1%}
- Details: {explanation}
- Note: This analysis is provided by the CNN-based XAI model
"""


# ============================================================================
# AGENTIC REASONER
# ============================================================================

class AgenticReasoner:
    """Handles reasoning, response generation, and MCP triggering"""
    
    def __init__(self, retriever: PubMedBERTRetriever, config: RAGConfig):
        self.retriever = retriever
        self.config = config
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.groq_client = Groq(api_key=api_key)
    
    def should_trigger_mcp(self, retrieved_docs: List[Dict]) -> bool:
        """
        Decide if MCP workflow should be triggered for external search
        
        Triggers MCP if:
        - No documents retrieved
        - All documents have low confidence
        - Query about recent research
        """
        if not retrieved_docs:
            return True
        
        avg_confidence = np.mean([doc['score'] for doc in retrieved_docs])
        if avg_confidence < self.config.confidence_threshold:
            return True
        
        return False
    
    def assess_complexity(self, query: str) -> float:
        """Assess query complexity (0-1 scale)"""
        factors = {
            'length': min(len(query.split()) / 20, 1.0),
            'questions': min(query.count('?') * 0.2, 0.4),
            'multiple_topics': 0.3 if ' and ' in query.lower() or ' or ' in query.lower() else 0.0,
            'specificity': 0.3 if any(word in query.lower() 
                                     for word in ['specific', 'exactly', 'compare', 'difference']) else 0.0
        }
        return min(sum(factors.values()), 1.0)
    
    def generate_response(
        self, 
        query: str, 
        context_docs: List[Dict], 
        query_type: QueryType,
        user_type: UserType,
        patient_context: Optional[PatientContext] = None
    ) -> str:
        """Generate personalized response based on user type"""
        
        context = self._build_context(context_docs)
        system_prompt = self._get_system_prompt(query_type, user_type)
        user_prompt = self._build_user_prompt(query, context, patient_context)
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _build_context(self, docs: List[Dict]) -> str:
        """Build context from retrieved documents"""
        if not docs:
            return "No relevant local documents found."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"[Source {i}] (Relevance: {doc['score']:.2%})\n{doc['text'][:500]}...\n"
            )
        return "\n".join(context_parts)
    
    def _get_system_prompt(self, query_type: QueryType, user_type: UserType) -> str:
        """Get personalized system prompt"""
        
        base_prompts = {
            UserType.DOCTOR: (
                "You are an AI assistant specialized in Alzheimer's disease research and clinical practice. "
                "Provide evidence-based, scientifically rigorous responses using medical terminology. "
                "Reference relevant studies, clinical guidelines, and current treatment protocols. "
                "Be precise and technical in your language."
            ),
            UserType.PATIENT: (
                "You are a compassionate AI assistant helping patients understand Alzheimer's disease. "
                "Use clear, simple language without medical jargon. Explain concepts in an accessible way. "
                "Be empathetic and supportive while providing accurate information. "
                "Always encourage consultation with healthcare professionals."
            ),
            UserType.CAREGIVER: (
                "You are a supportive AI assistant for Alzheimer's caregivers. "
                "Provide practical advice and emotional support. Use clear language and focus on "
                "actionable guidance for daily care challenges. Be empathetic to caregiver stress."
            )
        }
        
        query_additions = {
            QueryType.TREATMENT_OPTIONS: " Focus on evidence-based treatments and clinical trials.",
            QueryType.SYMPTOM_INQUIRY: " Clearly describe symptoms and their progression.",
            QueryType.LIFESTYLE_ADVICE: " Provide actionable lifestyle recommendations.",
            QueryType.MRI_INTERPRETATION: " Explain neuroimaging findings clearly.",
            QueryType.RESEARCH_UPDATE: " Present latest research with appropriate scientific context."
        }
        
        prompt = base_prompts.get(user_type, base_prompts[UserType.PATIENT])
        prompt += query_additions.get(query_type, "")
        
        return prompt
    
    def _build_user_prompt(
        self, 
        query: str, 
        context: str, 
        patient_context: Optional[PatientContext]
    ) -> str:
        """Build user prompt with all available context"""
        
        prompt_parts = ["Based on the following information, answer the user's question.\n"]
        
        # Add scientific sources
        prompt_parts.append(f"SCIENTIFIC SOURCES:\n{context}\n")
        
        # Add patient context if available
        if patient_context:
            prompt_parts.append(f"PATIENT CONTEXT:\n{patient_context.to_context_string()}\n")
        
        # Add query
        prompt_parts.append(f"USER QUESTION:\n{query}\n")
        
        # Add instructions
        prompt_parts.append(
            "\nINSTRUCTIONS:\n"
            "1. Answer based on provided sources and patient context\n"
            "2. If patient context includes MRI results, integrate this into your response\n"
            "3. Be accurate, clear, and appropriately personalized\n"
            "4. Always recommend consulting healthcare professionals\n"
            "5. If information is limited, acknowledge it honestly\n\n"
            "ANSWER:"
        )
        
        return "\n".join(prompt_parts)


# ============================================================================
# MAIN RAG SYSTEM
# ============================================================================

class AlzheimerRAGSystem:
    """Complete intelligent RAG system with XAI and personalization"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        
        print("\n" + "="*70)
        print("🧠 ALZHEIMER'S INTELLIGENT RAG SYSTEM WITH XAI")
        print("="*70)
        
        # Initialize components
        self.retriever = PubMedBERTRetriever(self.config)
        self.reasoner = AgenticReasoner(self.retriever, self.config)
        self.analyzer = QueryAnalyzer()
        self.explainer = XAIExplainer()
        self.mri_analyzer = MRIAnalyzer()  # Placeholder for CNN model
        
        print("✓ System initialized successfully!\n")
    
    def query(
        self,
        user_query: str,
        user_type: UserType = UserType.PATIENT,
        patient_context: Optional[PatientContext] = None,
        explain: bool = True
    ) -> Dict:
        """
        Main query interface
        
        Args:
            user_query: The user's question
            user_type: Doctor, patient, or caregiver
            patient_context: Optional patient information
            explain: Whether to include XAI explanations
        
        Returns:
            Dictionary with answer, evidence, explanations, and MCP status
        """
        
        print(f"\n{'='*70}")
        print(f"📝 QUERY: {user_query}")
        print(f"👤 USER TYPE: {user_type.value}")
        print(f"{'='*70}\n")
        
        # Step 1: Query analysis
        print("Step 1: Analyzing query...")
        query_type = self.analyzer.classify_query(user_query)
        entities = self.analyzer.extract_entities(user_query)
        complexity = self.reasoner.assess_complexity(user_query)
        print(f"  - Type: {query_type.value}")
        print(f"  - Complexity: {complexity:.2f}")
        print(f"  - Entities: {entities}")
        
        # Step 2: Document retrieval
        print("\nStep 2: Retrieving relevant documents...")
        top_k = 8 if complexity > 0.7 else self.config.top_k_retrieval
        retrieved_docs = self.retriever.retrieve(user_query, top_k=top_k)
        
        # Filter by confidence
        filtered_docs = [
            doc for doc in retrieved_docs 
            if doc['score'] >= self.config.confidence_threshold
        ]
        
        if not filtered_docs and retrieved_docs:
            filtered_docs = retrieved_docs[:3]
        
        print(f"  - Retrieved: {len(retrieved_docs)} documents")
        print(f"  - After filtering: {len(filtered_docs)} documents")
        
        # Step 3: Check if MCP should be triggered
        should_trigger_mcp = self.reasoner.should_trigger_mcp(filtered_docs)
        print(f"\nStep 3: MCP Decision: {'TRIGGER' if should_trigger_mcp else 'NOT NEEDED'}")
        
        # Step 4: Generate response
        print("\nStep 4: Generating personalized response...")
        answer = self.reasoner.generate_response(
            user_query, 
            filtered_docs, 
            query_type, 
            user_type,
            patient_context
        )
        
        # Step 5: Generate explanations
        explanations = None
        if explain:
            print("\nStep 5: Generating XAI explanations...")
            retrieval_exp = self.explainer.explain_retrieval(filtered_docs, user_query)
            
            explanations = {
                'retrieval': retrieval_exp,
                'mri': self.explainer.explain_mri_integration(
                    patient_context.mri_result if patient_context else None
                )
            }
        
        # Compile result
        result = {
            'answer': answer,
            'query_type': query_type.value,
            'user_type': user_type.value,
            'entities': entities,
            'evidence': filtered_docs,
            'confidence': filtered_docs[0]['score'] if filtered_docs else 0.0,
            'should_trigger_mcp': should_trigger_mcp,
            'mcp_reason': 'Low confidence in local documents' if should_trigger_mcp else None,
            'explanations': explanations,
            'patient_context_used': patient_context is not None
        }
        
        self._print_result(result)
        
        return result
    
    def _print_result(self, result: Dict):
        """Pretty print results"""
        print("\n" + "="*70)
        print("✨ RESPONSE")
        print("="*70)
        print(f"\n{result['answer']}")
        
        if result['explanations']:
            print(f"\n{result['explanations']['mri']}")
            
            print(f"\n\n CONFIDENCE METRICS:")
            ret_exp = result['explanations']['retrieval']
            print(f"  - Overall: {ret_exp['confidence']:.1%}")
            print(f"  - Top Score: {ret_exp['top_score']:.1%}")
            for reason in ret_exp['reasoning']:
                print(f"  {reason}")
            
            if ret_exp.get('suggestion'):
                print(f"\💡 SUGGESTION: {ret_exp['suggestion']}")
        
        if result['should_trigger_mcp']:
            print(f"\n MCP WORKFLOW TRIGGERED")
            print(f"   Reason: {result['mcp_reason']}")
            print(f"   Action: Searching external sources (PubMed, etc.)")
        
        



if __name__ == "__main__":
    # Initialize system
    config = RAGConfig(
        pdf_dir=PDF_DIR,
        output_dir=OUT_DIR
    )
    
    rag_system = AlzheimerRAGSystem(config)
    
    # Example 1: Patient query with context
    
    print("\n\nEXAMPLE 1: Patient Query with MRI Results")
    
    
    patient_context = PatientContext(
        age=68,
        symptoms=['memory loss', 'confusion', 'difficulty with daily tasks'],
        mri_result={
            'prediction': 'mild_cognitive_impairment',
            'confidence': 0.78,
            'explanation': 'Mild hippocampal atrophy detected'
        },
        stage=AlzheimerStage.MILD_IMPAIRMENT
    )
    
    result1 = rag_system.query(
        "MY DOCTOR TOLD ME that i have alzheimer what is it?",
        user_type=UserType.PATIENT,
        patient_context=patient_context,
        explain=True
    )
    
    # Example 2: Doctor query
   
    #print("\n\nEXAMPLE 2: Doctor Query")
  
    
    #result2 = rag_system.query(
       # "What are the latest clinical trial results for anti-amyloid antibodies?",
      #  user_type=UserType.DOCTOR,
      #  explain=True
   # )
    
   # print("\n\nSystem ready for FastAPI integration!")