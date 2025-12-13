"""
Knowledge Graph Integration for Alzheimer's RAG
Uses NetworkX for in-memory graph (no external database needed)
Can be upgraded to Neo4j for production
"""

import networkx as nx
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import re
from app.config import OUT_DIR


@dataclass
class Entity:
    """Medical entity in knowledge graph"""
    id: str
    type: str  # symptom, biomarker, treatment, test, stage, risk_factor
    name: str
    properties: Dict


@dataclass
class Relationship:
    """Relationship between entities"""
    source: str  # entity id
    target: str  # entity id
    relation_type: str  # causes, treats, indicates, diagnosed_by, etc.
    properties: Dict


class MedicalKnowledgeGraph:
    """
    Knowledge graph for Alzheimer's disease
    Stores entities and relationships for reasoning
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # Directed graph with multiple edges
        self._initialize_base_knowledge()
        
        # Load dynamic knowledge if available
        relations_path = Path(OUT_DIR) / "relations.json"
        if relations_path.exists():
            self.load_from_json(str(relations_path))
            print(f"✓ Loaded dynamic knowledge from {relations_path.name}")
    
    def _initialize_base_knowledge(self):
        """Initialize with core Alzheimer's knowledge"""
        
        # Cognitive Tests
        self.add_entity(Entity(
            id="test_mmse",
            type="test",
            name="MMSE",
            properties={
                "full_name": "Mini-Mental State Examination",
                "max_score": 30,
                "normal_range": "24-30"
            }
        ))
        
        self.add_entity(Entity(
            id="test_moca",
            type="test",
            name="MoCA",
            properties={
                "full_name": "Montreal Cognitive Assessment",
                "max_score": 30,
                "normal_range": "26-30"
            }
        ))
        
        # Disease Stages
        stages = [
            ("stage_normal", "Normal Cognition", {"mmse_range": "24-30", "moca_range": "26-30"}),
            ("stage_mci", "Mild Cognitive Impairment", {"mmse_range": "20-24", "moca_range": "18-26"}),
            ("stage_mild_ad", "Mild Alzheimer's", {"mmse_range": "20-24", "moca_range": "10-20"}),
            ("stage_moderate_ad", "Moderate Alzheimer's", {"mmse_range": "10-20", "moca_range": "5-15"}),
            ("stage_severe_ad", "Severe Alzheimer's", {"mmse_range": "0-10", "moca_range": "0-10"})
        ]
        
        for stage_id, stage_name, props in stages:
            self.add_entity(Entity(
                id=stage_id,
                type="stage",
                name=stage_name,
                properties=props
            ))
        
        # Symptoms
        symptoms = [
            ("symptom_memory_loss", "Memory Loss", {"severity": "variable"}),
            ("symptom_confusion", "Confusion", {"severity": "variable"}),
            ("symptom_language_difficulty", "Language Difficulty", {"severity": "variable"}),
            ("symptom_disorientation", "Disorientation", {"severity": "variable"}),
            ("symptom_mood_changes", "Mood Changes", {"severity": "variable"})
        ]
        
        for symptom_id, symptom_name, props in symptoms:
            self.add_entity(Entity(
                id=symptom_id,
                type="symptom",
                name=symptom_name,
                properties=props
            ))
        
        # Biomarkers
        biomarkers = [
            ("biomarker_amyloid", "Amyloid Beta", {"type": "protein"}),
            ("biomarker_tau", "Tau Protein", {"type": "protein"}),
            ("biomarker_apoe4", "APOE4 Gene", {"type": "genetic"})
        ]
        
        for bio_id, bio_name, props in biomarkers:
            self.add_entity(Entity(
                id=bio_id,
                type="biomarker",
                name=bio_name,
                properties=props
            ))
        
        # Treatments
        treatments = [
            ("treatment_donepezil", "Donepezil", {"type": "cholinesterase_inhibitor", "stage": "mild_to_moderate"}),
            ("treatment_memantine", "Memantine", {"type": "nmda_antagonist", "stage": "moderate_to_severe"}),
            ("treatment_rivastigmine", "Rivastigmine", {"type": "cholinesterase_inhibitor", "stage": "mild_to_moderate"}),
            ("treatment_galantamine", "Galantamine", {"type": "cholinesterase_inhibitor", "stage": "mild_to_moderate"})
        ]
        
        for treat_id, treat_name, props in treatments:
            self.add_entity(Entity(
                id=treat_id,
                type="treatment",
                name=treat_name,
                properties=props
            ))
        
        # Risk Factors
        risk_factors = [
            ("risk_age", "Advanced Age", {"modifiable": False}),
            ("risk_apoe4", "APOE4 Gene", {"modifiable": False}),
            ("risk_cardiovascular", "Cardiovascular Disease", {"modifiable": True}),
            ("risk_diabetes", "Diabetes", {"modifiable": True}),
            ("risk_sedentary", "Sedentary Lifestyle", {"modifiable": True})
        ]
        
        for risk_id, risk_name, props in risk_factors:
            self.add_entity(Entity(
                id=risk_id,
                type="risk_factor",
                name=risk_name,
                properties=props
            ))
        
        # Add relationships
        self._add_base_relationships()
    
    def _add_base_relationships(self):
        """Add relationships between entities"""
        
        # Tests diagnose stages
        self.add_relationship(Relationship(
            source="test_mmse",
            target="stage_normal",
            relation_type="indicates",
            properties={"score_range": "24-30"}
        ))
        
        self.add_relationship(Relationship(
            source="test_mmse",
            target="stage_mci",
            relation_type="indicates",
            properties={"score_range": "20-24"}
        ))
        
        self.add_relationship(Relationship(
            source="test_moca",
            target="stage_normal",
            relation_type="indicates",
            properties={"score_range": "26-30"}
        ))
        
        # Symptoms indicate stages
        self.add_relationship(Relationship(
            source="symptom_memory_loss",
            target="stage_mci",
            relation_type="suggests",
            properties={"confidence": "moderate"}
        ))
        
        self.add_relationship(Relationship(
            source="symptom_confusion",
            target="stage_mild_ad",
            relation_type="suggests",
            properties={"confidence": "high"}
        ))
        
        # Biomarkers indicate disease
        self.add_relationship(Relationship(
            source="biomarker_amyloid",
            target="stage_mild_ad",
            relation_type="indicates",
            properties={"when": "elevated"}
        ))
        
        self.add_relationship(Relationship(
            source="biomarker_tau",
            target="stage_mild_ad",
            relation_type="indicates",
            properties={"when": "elevated"}
        ))
        
        # Treatments for stages
        self.add_relationship(Relationship(
            source="treatment_donepezil",
            target="stage_mild_ad",
            relation_type="treats",
            properties={"efficacy": "moderate"}
        ))
        
        self.add_relationship(Relationship(
            source="treatment_memantine",
            target="stage_moderate_ad",
            relation_type="treats",
            properties={"efficacy": "moderate"}
        ))
        
        # Risk factors increase risk
        self.add_relationship(Relationship(
            source="risk_apoe4",
            target="stage_mild_ad",
            relation_type="increases_risk",
            properties={"fold_increase": "3-4x"}
        ))
    
    def add_entity(self, entity: Entity):
        """Add entity to graph"""
        self.graph.add_node(
            entity.id,
            entity_type=entity.type,
            name=entity.name,
            **entity.properties
        )
    
    def add_relationship(self, relationship: Relationship):
        """Add relationship to graph"""
        self.graph.add_edge(
            relationship.source,
            relationship.target,
            relation_type=relationship.relation_type,
            **relationship.properties
        )
    
    def find_paths(
        self, 
        start_entity: str, 
        end_entity: str, 
        max_length: int = 3
    ) -> List[List[str]]:
        """Find all paths between two entities"""
        try:
            paths = list(nx.all_simple_paths(
                self.graph, 
                start_entity, 
                end_entity, 
                cutoff=max_length
            ))
            return paths
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []
    
    def get_neighbors(
        self, 
        entity_id: str, 
        relation_type: Optional[str] = None
    ) -> List[Dict]:
        """Get neighboring entities"""
        if entity_id not in self.graph:
            return []
        
        neighbors = []
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, neighbor)
            
            # Filter by relation type if specified
            if relation_type:
                matching_edges = [
                    edge for edge in edge_data.values() 
                    if edge.get('relation_type') == relation_type
                ]
                if not matching_edges:
                    continue
            
            node_data = self.graph.nodes[neighbor]
            neighbors.append({
                'id': neighbor,
                'type': node_data.get('entity_type'),
                'name': node_data.get('name'),
                'properties': {k: v for k, v in node_data.items() if k not in ['entity_type', 'name']}
            })
        
        return neighbors
    
    def reason_about_patient(self, patient_context: Dict) -> Dict:
        """
        Reason about patient using knowledge graph
        Returns insights and connections
        """
        insights = {
            'stage_indicators': [],
            'relevant_treatments': [],
            'risk_factors': [],
            'reasoning_paths': []
        }
        
        # Analyze MMSE score
        if patient_context.get('mmse_score') is not None:
            mmse = patient_context['mmse_score']
            
            # Find indicated stage
            if mmse >= 24:
                stage = "stage_normal"
                stage_name = "Normal Cognition"
            elif mmse >= 20:
                stage = "stage_mci"
                stage_name = "Mild Cognitive Impairment"
            elif mmse >= 10:
                stage = "stage_moderate_ad"
                stage_name = "Moderate Alzheimer's"
            else:
                stage = "stage_severe_ad"
                stage_name = "Severe Alzheimer's"
            
            insights['stage_indicators'].append({
                'source': 'MMSE',
                'score': mmse,
                'indicates': stage_name,
                'confidence': 'moderate'
            })
            
            # Find treatments for this stage
            treatments = self.get_neighbors(stage, relation_type='treats')
            # Reverse direction - find what treats this stage
            for node_id in self.graph.nodes():
                if self.graph.nodes[node_id].get('entity_type') == 'treatment':
                    if self.graph.has_edge(node_id, stage):
                        edge_data = self.graph.get_edge_data(node_id, stage)
                        for edge in edge_data.values():
                            if edge.get('relation_type') == 'treats':
                                treatment_data = self.graph.nodes[node_id]
                                insights['relevant_treatments'].append({
                                    'name': treatment_data.get('name'),
                                    'type': treatment_data.get('type'),
                                    'efficacy': edge.get('efficacy', 'unknown')
                                })
        
        # Analyze MoCA score
        if patient_context.get('moca_score') is not None:
            moca = patient_context['moca_score']
            
            if moca >= 26:
                stage_name = "Normal"
            else:
                stage_name = "Possible Cognitive Impairment"
            
            insights['stage_indicators'].append({
                'source': 'MoCA',
                'score': moca,
                'indicates': stage_name,
                'confidence': 'high'
            })
        
        # Analyze biomarkers
        if patient_context.get('biomarkers'):
            for biomarker, value in patient_context['biomarkers'].items():
                if value != 'unknown' and value != 'negative':
                    biomarker_id = f"biomarker_{biomarker.lower()}"
                    if biomarker_id in self.graph:
                        # Find what this biomarker indicates
                        indicated = self.get_neighbors(biomarker_id, relation_type='indicates')
                        for indication in indicated:
                            insights['stage_indicators'].append({
                                'source': biomarker,
                                'value': value,
                                'indicates': indication['name'],
                                'confidence': 'high'
                            })
        
        # Analyze age as risk factor
        if patient_context.get('age'):
            age = patient_context['age']
            if age >= 65:
                insights['risk_factors'].append({
                    'factor': 'Advanced Age',
                    'value': age,
                    'modifiable': False,
                    'impact': 'high' if age >= 75 else 'moderate'
                })
        
        return insights
    
    def get_explanation_path(
        self, 
        symptom: str, 
        treatment: str
    ) -> Optional[str]:
        """Get natural language explanation of path from symptom to treatment"""
        
        # Find symptom entity
        symptom_id = None
        for node_id, data in self.graph.nodes(data=True):
            if data.get('entity_type') == 'symptom' and symptom.lower() in data.get('name', '').lower():
                symptom_id = node_id
                break
        
        # Find treatment entity
        treatment_id = None
        for node_id, data in self.graph.nodes(data=True):
            if data.get('entity_type') == 'treatment' and treatment.lower() in data.get('name', '').lower():
                treatment_id = node_id
                break
        
        if not symptom_id or not treatment_id:
            return None
        
        # Find path
        paths = self.find_paths(symptom_id, treatment_id, max_length=3)
        
        if not paths:
            return None
        
        # Build explanation from first path
        path = paths[0]
        explanation_parts = []
        
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            source_name = self.graph.nodes[source].get('name')
            target_name = self.graph.nodes[target].get('name')
            
            edge_data = self.graph.get_edge_data(source, target)
            relation = list(edge_data.values())[0].get('relation_type', 'relates to')
            
            explanation_parts.append(f"{source_name} {relation} {target_name}")
        
        return " → ".join(explanation_parts)
    
    def save(self, filepath: str):
        """Save graph to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f)
    
        with open(filepath, 'rb') as f:
            self.graph = pickle.load(f)

    def load_from_json(self, filepath: str):
        """Load relations from JSON file and merge into graph"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                relations = json.load(f)
            
            count = 0
            for r in relations:
                # Add nodes if they don't exist
                if r['source'] not in self.graph:
                    self.graph.add_node(r['source'], name=r['source'], entity_type='extracted')
                if r['target'] not in self.graph:
                    self.graph.add_node(r['target'], name=r['target'], entity_type='extracted')
                
                # Add edge
                self.graph.add_edge(
                    r['source'],
                    r['target'],
                    relation_type=r['relation'],
                    confidence=r.get('confidence', 0.8),
                    source_doc=r.get('document', 'unknown')
                )
                count += 1
            
            print(f"✓ Merged {count} dynamic relations into Knowledge Graph")
            
        except Exception as e:
            print(f"⚠️  Could not load dynamic relations: {e}")


class GraphEnhancedRetriever:
    """
    Combines vector retrieval with knowledge graph reasoning
    """
    
    def __init__(self, vector_retriever, knowledge_graph: MedicalKnowledgeGraph):
        self.vector_retriever = vector_retriever
        self.kg = knowledge_graph
    
    def retrieve_with_graph(
        self,
        query: str,
        patient_context: Optional[Dict] = None,
        top_k: int = 8
    ) -> Tuple[List[Dict], Dict]:
        """
        Hybrid retrieval: vector search + knowledge graph reasoning
        """
        
        # Step 1: Vector retrieval
        vector_results, vector_metrics = self.vector_retriever.retrieve(
            query=query,
            top_k=top_k
        )
        
        # Step 2: Knowledge graph reasoning
        graph_insights = {}
        if patient_context:
            graph_insights = self.kg.reason_about_patient(patient_context)
        
        # Step 3: Enhance results with graph insights
        enhanced_results = vector_results.copy()
        
        # Add graph-based context to results
        for result in enhanced_results:
            result['graph_insights'] = graph_insights
        
        # Step 4: Boost results that match graph insights
        if graph_insights.get('stage_indicators'):
            indicated_stages = [ind['indicates'] for ind in graph_insights['stage_indicators']]
            
            for result in enhanced_results:
                text_lower = result['text'].lower()
                # Boost if document mentions indicated stages
                for stage in indicated_stages:
                    if stage.lower() in text_lower:
                        result['score'] *= 1.2  # 20% boost
        
        # Re-sort by boosted scores
        enhanced_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Combine metrics
        combined_metrics = {
            **vector_metrics,
            'graph_insights': graph_insights,
            'graph_enhanced': True
        }
        
        return enhanced_results[:top_k], combined_metrics


# Convenience function
def create_knowledge_graph() -> MedicalKnowledgeGraph:
    """Create and return initialized knowledge graph"""
    return MedicalKnowledgeGraph()


def create_graph_enhanced_retriever(vector_retriever) -> GraphEnhancedRetriever:
    """Create graph-enhanced retriever"""
    kg = create_knowledge_graph()
    return GraphEnhancedRetriever(vector_retriever, kg)
