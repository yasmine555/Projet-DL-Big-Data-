from typing import TypedDict, Annotated, Sequence
import operator
import os
import json
import re
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from openai import OpenAI

from app.services.agent_tools import (
    get_patient_records,
    search_medical_docs,
    query_knowledge_graph,
    web_search_n8n,
    fetch_latest_research,
)

# -------------------------------------------------
# Setup
# -------------------------------------------------
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# -------------------------------------------------
# Utilities
# -------------------------------------------------
P_GREETING = re.compile(r"\b(hi|hello|hey|greetings|morning|afternoon|evening)\b", re.IGNORECASE)
P_GOODBYE = re.compile(r"\b(bye|goodbye|see you|later|cya)\b", re.IGNORECASE)
P_THANKS = re.compile(r"\b(thanks|thank you|thx|appreciate)\b", re.IGNORECASE)

# -------------------------------------------------
# State
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    original_query: str
    user_role: str
    patient_id: str
    context_type: str
    retry_count: int
    quality_score: float
    last_tool_used: str | None
    citations: list


# -------------------------------------------------
# Agent Orchestrator
# -------------------------------------------------
class AgentOrchestrator:
    def __init__(self):
        self.ROUTER_MODEL = "allenai/olmo-3.1-32b-think:free"
        self.GENERATOR_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
        self.EVALUATOR_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"

        self.tools = [
            get_patient_records,
            search_medical_docs,
            query_knowledge_graph,
            web_search_n8n,
            fetch_latest_research,
        ]

        self.tool_node = ToolNode(self.tools)
        self.graph = self._build_graph()

    # -------------------------------------------------
    # Graph
    # -------------------------------------------------
    def _build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("router", self.node_router)
        g.add_node("tools", self.node_tools)
        g.add_node("generator", self.node_generator)
        g.add_node("evaluator", self.node_evaluator)

        g.set_entry_point("router")

        g.add_conditional_edges(
            "router",
            self.route_decision,
            {
                "tools": "tools",
                "generate": "generator",
                "end": END,
            },
        )

        g.add_edge("tools", "generator")
        g.add_edge("generator", "evaluator")

        g.add_conditional_edges(
            "evaluator",
            self.eval_decision,
            {"approved": END, "retry": "router"},
        )

        return g.compile()

    # -------------------------------------------------
    # Router
    # -------------------------------------------------
    def node_router(self, state: AgentState):
        user_msg = state["original_query"]
        print(f"\n--- [STEP: ROUTER] Analyzing query: '{user_msg}' ---")

        # 🚀 FAST EXIT: Specific responses
        if P_GREETING.search(user_msg):
            return {
                "messages": [AIMessage(content="Hello! I am your AI Medical Assistant. How can I help you regarding this patient today?")],
                "last_tool_used": None,
            }
        
        if P_GOODBYE.search(user_msg):
            return {
                "messages": [AIMessage(content="Goodbye! Take care.")],
                "last_tool_used": None,
            }

        if P_THANKS.search(user_msg):
            return {
                "messages": [AIMessage(content="You're welcome! Let me know if you need anything else.")],
                "last_tool_used": None,
            }

        role = state["user_role"]
        context_type = state["context_type"]

        if context_type == "summary":
            system_prompt = """
You are a medical routing agent.
Always call get_patient_records for patient summaries.

Respond ONLY in JSON:
{ "tool": "get_patient_records" }
"""
        else:
            system_prompt = f"""
You are a medical routing agent.

User role: {role}

Choose relevant tools (you can select multiple):
- get_patient_records → for ANY questions about "this patient", "the patient", "my results", "medical history", "risks", "diagnosis", "analysis", or "summary".
- search_medical_docs → medical facts, drugs, treatments, diseases, guidelines , risks , activities , recommendations , symptoms, tests, etc.
- query_knowledge_graph → relationships between diseases, genes, proteins , symptoms
- fetch_latest_research → LATEST clinical trials, scientific papers, NIH/PubMed studies, new drug approvals.
- web_search_n8n → very recent news
- none → general answer

Respond ONLY in JSON:
{{ "tools": ["<tool_name>", "<tool_name>", "<tool_name>"] }}
"""

        res = client.chat.completions.create(
            model=self.ROUTER_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
        )

        content = res.choices[0].message.content.strip().replace("```json", "").replace("```", "")

        try:
            decision = json.loads(content)
        except Exception:
            decision = {"tools": []}

        tools = decision.get("tools", [])
        # Fallback for old single-tool format if model hallucinates
        if "tool" in decision and decision["tool"] != "none":
             tools.append(decision["tool"])

        # Deduplicate and clean
        tools = [t for t in tools if t != "none"]
        
        tool_str = ",".join(tools)
        print(f"--- [STEP: ROUTER] Decision: Use Tools '{tool_str}' ---")

        if tools:
            return {
                "messages": [AIMessage(content=f"TOOL_CALL::{tool_str}")],
                "last_tool_used": tool_str,
            }

        return {
            "messages": [AIMessage(content="DIRECT_GENERATION")],
            "last_tool_used": None,
        }

    def route_decision(self, state: AgentState):
        last = state["messages"][-1].content
        if last.startswith("TOOL_CALL::"):
            return "tools"
        if last.startswith("DIRECT_GENERATION"):
            return "generate"
        return "end"

    # -------------------------------------------------
    # Tools
    # -------------------------------------------------
    async def node_tools(self, state: AgentState):
        last_msg = state["messages"][-1].content
        tool_str = last_msg.replace("TOOL_CALL::", "")
        tool_names = [t.strip() for t in tool_str.split(",") if t.strip()]
        
        patient_id = state.get("patient_id", "unknown")
        query = state["original_query"]
        
        combined_output = []

        for tool_name in tool_names:
            print(f"\n--- [STEP: TOOLS] Executing Tool: {tool_name} (Patient ID: {patient_id}) ---")
            
            output = "Tool output missing."
            try:
                if tool_name == "get_patient_records":
                    output = await get_patient_records(patient_id)
                elif tool_name == "search_medical_docs":
                    output = search_medical_docs(query)
                elif tool_name == "query_knowledge_graph":
                    output = query_knowledge_graph(query)
                elif tool_name == "web_search_n8n":
                    output = await web_search_n8n(query)
                elif tool_name == "fetch_latest_research":
                    output = await fetch_latest_research(query)
                else:
                    output = f"Unknown tool: {tool_name}"
            except Exception as e:
                output = f"Error executing {tool_name}: {e}"
            
            combined_output.append(f"### Tool Output ({tool_name}):\n{output}")

        final_content = "\n\n".join(combined_output)

        return {
            "messages": [
                HumanMessage(content=query),
                AIMessage(content=final_content),
            ],
            "last_tool_used": tool_str,
        }

    # -------------------------------------------------
    # Generator
    # -------------------------------------------------
    def node_generator(self, state: AgentState):
        print("\n--- [STEP: GENERATOR] Synthesizing answer from context... ---")
        role = state["user_role"]
        
        if role == "patient":
            persona = (
                "a friendly, reassuring medical guide who explains health information using SIMPLE, everyday language (5th-grade level). "
                "You are encouraging and supportive. You avoid complex medical jargon completely."
            )
            instructions = """
            1. **Simple Vocabulary**: Use "memory loss" instead of "cognitive decline", "brain scan" instead of "neuroimaging", etc.
            2. **Be Encouraging**: Start with a supportive tone. If results are normal, emphasize that. If there are risks, frame them as "things to watch" rather than "diagnoses".
            3. **Explain Context**: If the patient entered symptoms, acknowledge them directly (e.g., "You mentioned you feel forgetful...").
            4. **No Complexity**: Do NOT mention "biomarkers", "amyloid", "tau", "CSF", "MMSE scores", or "pathophysiology" unless you immediately explain them in very simple terms.
            5. **Direct Answer**: Answer the question directly.
            6. **Missing Data**: If tests (like MRI) are missing, just say "We don't have that visual information yet" instead of listing "neuroimaging findings: not provided".
            """
        else:
            persona = "a clinical decision support assistant using professional medical terminology."
            instructions = """
            1. **Synthesize**: Convert raw patient data into a professional clinical narrative.
            2. **Interpret**: Use available metrics to form a clinical picture (e.g., "MSE 24 suggests mild impairment").
            3. **Professional Tone**: Use precise medical terminology.
            4. **Context Usage**: Use 'patient_record' and 'latest_analysis_result' to support your answer.
            """

        # Extract context and potential citations
        context_parts = []
        sources = []
        import json
        import re

        for m in state["messages"]:
            if isinstance(m, AIMessage) and "Tool Output" in m.content:
                tool_output = m.content.replace(f"Tool Output ({state.get('last_tool_used', 'unknown')}):\n", "")
                context_parts.append(m.content)
                
                # 1. Try JSON extraction first
                try:
                    data = json.loads(tool_output)
                    if isinstance(data, dict):
                        if data.get("source") == "database_dump":
                            sources.append("Patient Medical Record (Database)")
                        if "patient_record" in data:
                             sources.append("Patient Demographics & History")
                        if "latest_analysis_result" in data:
                             sources.append(f"MRI Analysis & Questionnaire Result ({data.get('latest_analysis_result', {}).get('created_at', 'Unknown Date')})")
                except json.JSONDecodeError:
                    pass

                # 2. Text-based extraction (Regex) for RAG docs
                found_sources = re.findall(r'(?:Source|Title):\s*(.+)', tool_output, re.IGNORECASE)
                sources.extend(found_sources)
                
                # 3. Fallback naive line check
                for line in tool_output.split('\n'):
                    if line.strip().startswith("Source:") or line.strip().startswith("Title:"):
                        clean_source = line.split(":", 1)[1].strip()
                        sources.append(clean_source)

        context = "\n\n".join(context_parts)
        if not context:
             context = "No specific external information found. Answer based on general medical knowledge."

        prompt = f"""
You are {persona}.

**Instructions:**
{instructions}

**Data Availability Rule:**
- **STRICTLY NO HALLUCINATIONS**: Do NOT mention specific scores (like PHQ-9) unless they are in the context.

**Context:**
{context}

**User Query:** {state["original_query"]}
"""

        res = client.chat.completions.create(
            model=self.GENERATOR_MODEL,
            temperature=0.2,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": prompt}
            ],
        )
        
        final_answer = res.choices[0].message.content
        
        # Append sources if found
        if sources:
             unique_sources = list(set(sources))
             final_answer += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in unique_sources])

        return {
            "messages": [AIMessage(content=final_answer)]
        }

    # -------------------------------------------------
    # Evaluator
    # -------------------------------------------------
    def node_evaluator(self, state: AgentState):
        answer = state["messages"][-1].content
        original_query = state["original_query"]
        # Extract only tool outputs for context
        context = "\n".join([m.content for m in state["messages"] if isinstance(m, AIMessage) and "Tool Output" in m.content])

        prompt = f"""
Evaluate the following answer to the query.
Score from 0.0 to 1.0 (float).

Criteria:
1. Grounding: Is the answer supported by the provided Context?
2. Relevance: Does it answer the Query?
3. Safety: Is it medically safe?

Query: {original_query}
Context snippets: {context[:1000]}...
Answer: {answer}

Output ONLY the float score.
"""

        res = client.chat.completions.create(
            model=self.EVALUATOR_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        try:
            score = float(res.choices[0].message.content.strip())
        except Exception:
            score = 0.8  # Default to passing on error

        # Increment retry count
        new_count = state.get("retry_count", 0) + 1
        return {"quality_score": score, "retry_count": new_count}

    def eval_decision(self, state: AgentState):
        score = state.get("quality_score", 0)
        retries = state.get("retry_count", 0)
        
        # Threshold 0.7, but MAX 1 retry to prevent latency issues
        if score >= 0.7 or retries > 1:
            return "approved"
        return "retry"

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------
    async def run(
        self,
        query: str,
        user_role: str,
        patient_id: str,
        context_type: str = "chatbot",
    ):
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "original_query": query,
            "user_role": user_role,
            "patient_id": patient_id,
            "context_type": context_type,
            "retry_count": 0,
            "quality_score": 0.0,
            "last_tool_used": None,
            "citations": [],
        }

        final = await self.graph.ainvoke(initial_state)
        return final["messages"][-1].content
