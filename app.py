import streamlit as st
import asyncio
import time
import numpy as np
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# ==========================================
# 0. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="AegisAI Workspace",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom UI CSS injection for polished aesthetics
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        border-bottom: 2px solid #2563eb !important;
    }
    div.stButton > button {
        border-radius: 6px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 1. CORE ENGINE BACKENDS
# ==========================================

class ProductionHybridEngine:
    """Hybrid Retrieval Engine (Dense Vector + BM25 Lexical + RRF)."""
    def __init__(self):
        self.documents = [
            {"id": "doc-01", "title": "Hybrid Search Architecture", "text": "AegisAI utilizes Reciprocal Rank Fusion (RRF) to seamlessly merge dense semantic embeddings with BM25 lexical keyword matching."},
            {"id": "doc-02", "title": "Microservice Resilience", "text": "Enterprise LLM setups require circuit breakers, exponential backoff retries, and automated provider failover layers."},
            {"id": "doc-03", "title": "Synthetic Data Pipelines", "text": "Automated LLM-as-a-Judge gates validate synthesized instruction-tuning records for faithfulness before fine-tuning exports."}
        ]

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        # Simulated intelligent matching score distribution
        results = []
        for idx, doc in enumerate(self.documents):
            score = round(0.95 - (idx * 0.12) + np.random.uniform(-0.02, 0.02), 4)
            results.append({**doc, "rrf_score": score})
        return sorted(results, key=lambda x: x["rrf_score"], reverse=True)[:top_k]

class DatabaseQueryInput(BaseModel):
    sql_query: str = Field(..., description="Sanitized SQL query string.")

class AgentRuntimeEngine:
    """Isolated Agent Plugin Runtime."""
    def execute_tool(self, tool_name: str, query: str) -> Dict[str, Any]:
        # Validate using pydantic schema model
        validated = DatabaseQueryInput(sql_query=query)
        time.sleep(0.18) # Simulate processing latency
        return {
            "status": "success",
            "tool": tool_name,
            "latency_ms": 142.5,
            "payload_executed": validated.sql_query,
            "rows_affected": 2,
            "result_data": [{"id": 101, "role": "engineer", "status": "active"}, {"id": 102, "role": "architect", "status": "active"}]
        }

class ResilientRouterEngine:
    """Multi-Provider Gateway & Fallback Router."""
    def route_request(self, prompt: str, force_failover: bool) -> Dict[str, Any]:
        time.sleep(0.2)
        if force_failover:
            return {
                "provider": "Fallback-Anthropic (Claude 3.5 Sonnet)",
                "failover_triggered": True,
                "reason": "Primary endpoint returned HTTP 429 (Rate Limit Exceeded). Routing cleanly.",
                "response": f"Successfully synthesized response via failover route for: '{prompt}'"
            }
        return {
            "provider": "Primary-OpenAI (GPT-4o)",
            "failover_triggered": False,
            "reason": "Primary endpoint healthy.",
            "response": f"Successfully generated direct response for: '{prompt}'"
        }

class EvaluationEngine:
    """Automated Quality Gate Engine."""
    def evaluate(self, question: str, context: str, response: str) -> Dict[str, Any]:
        faithfulness = round(np.random.uniform(0.89, 0.98), 2)
        relevance = round(np.random.uniform(0.85, 0.96), 2)
        passed = faithfulness >= 0.90 and relevance >= 0.88
        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "passed": passed,
            "feedback": "Semantics align accurately with provided source context. Tone and structure conform to standard instructions."
        }


# ==========================================
# 2. SESSION STATE MANAGEMENT
# ==========================================
if "hybrid" not in st.session_state: st.session_state.hybrid = ProductionHybridEngine()
if "agent" not in st.session_state: st.session_state.agent = AgentRuntimeEngine()
if "router" not in st.session_state: st.session_state.router = ResilientRouterEngine()
if "evaluator" not in st.session_state: st.session_state.evaluator = EvaluationEngine()


# ==========================================
# 3. SIDEBAR NAVIGATION & CONTROLS
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("AegisAI Console")
    st.caption("v2.4.0-production")
    
    st.markdown("---")
    
    selected_module = st.radio(
        "Navigation Hub",
        [
            "🔍 Hybrid RAG Retriever",
            "⚡ Modular Agent Runtime",
            "🔀 Resilient LLM Gateway",
            "📊 Synthetic Quality Gates"
        ]
    )
    
    st.markdown("---")
    st.markdown("### System Status")
    st.markdown("🟢 **Vector Index:** Synced")
    st.markdown("🟢 **Plugin Runtime:** Active")
    st.markdown("🟢 **Gateway Proxy:** Optimal")


# ==========================================
# 4. MODULE VIEWS & UI INTERACTION
# ==========================================

# --- MODULE 1: HYBRID RAG ---
if selected_module == "🔍 Hybrid RAG Retriever":
    st.title("Hybrid Retrieval-Augmented Generation")
    st.markdown("Evaluate precision and recall performance using **Dense Embeddings + BM25 Lexical Matching** combined with **Reciprocal Rank Fusion (RRF)**.")
    
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Semantic / Keyword Query", value="How does AegisAI handle hybrid search?")
        with col2:
            top_k = st.selectbox("Top Results (K)", [1, 2, 3], index=1)
            
        if st.button("Execute Retrieval Pipeline", type="primary", use_container_width=True):
            with st.spinner("Executing hybrid search across sparse and dense indices..."):
                time.sleep(0.3)
                results = st.session_state.hybrid.search(search_query, top_k)
                
            st.success(f"Retrieved {len(results)} high-relevance contexts successfully.")
            
            for i, res in enumerate(results):
                with st.container(border=True):
                    c_left, c_right = st.columns([4, 1])
                    with c_left:
                        st.markdown(f"#### #{i+1} : {res['title']}")
                        st.write(res['text'])
                    with c_right:
                        st.metric("RRF Score", res['rrf_score'])
                        st.caption(f"ID: `{res['id']}`")


# --- MODULE 2: AGENT RUNTIME ---
elif selected_module == "⚡ Modular Agent Runtime":
    st.title("Sandboxed Plugin & Tool Runtime")
    st.markdown("Test secure tool execution decoupled from core agent logic, backed by **strict Pydantic input schemas** and **execution telemetry**.")
    
    with st.container(border=True):
        col1, col2 = st.columns([2, 2])
        with col1:
            tool_selection = st.selectbox("Select Available Plugin", ["db_query_plugin"])
        with col2:
            st.text_input("Validation Schema", value="DatabaseQueryInput", disabled=True)
            
        sql_input = st.text_area("Tool Argument Payload (`sql_query`)", value="SELECT id, role, status FROM users WHERE status = 'active';")
        
        if st.button("Dispatch Tool Securely", type="primary", use_container_width=True):
            with st.spinner("Validating parameters and executing in sandbox..."):
                try:
                    res = st.session_state.agent.execute_tool(tool_selection, sql_input)
                    st.success("Tool executed without errors.")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Execution Status", res["status"].upper())
                    m2.metric("Latency", f"{res['latency_ms']} ms")
                    m3.metric("Rows Affected", res["rows_affected"])
                    
                    st.markdown("#### Structured JSON Output")
                    st.json(res["result_data"])
                except Exception as ex:
                    st.error(f"Validation Error: {ex}")


# --- MODULE 3: RESILIENT GATEWAY ---
elif selected_module == "🔀 Resilient LLM Gateway":
    st.title("Multi-Provider Failover Gateway")
    st.markdown("Simulate upstream LLM infrastructure issues, rate limits (`429`), and provider outages to verify **automatic failover behavior**.")
    
    with st.container(border=True):
        prompt_text = st.text_area("Prompt Payload", value="Summarize the core benefits of distributed API gateways.")
        simulate_outage = st.checkbox("Simulate Primary Provider Outage (HTTP 429 / Rate Limit)", value=False)
        
        if st.button("Send Request through Gateway", type="primary", use_container_width=True):
            with st.spinner("Routing traffic through primary and secondary provider pools..."):
                result = st.session_state.router.route_request(prompt_text, simulate_outage)
                
            if result["failover_triggered"]:
                st.warning(f"⚠️ **Failover Activated:** {result['reason']}")
            else:
                st.info(f"✅ **Direct Routing:** {result['reason']}")
                
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric("Active Provider", result["provider"])
            with col_b:
                st.markdown("#### Gateway Output Response")
                st.write(result["response"])


# --- MODULE 4: SYNTHETIC EVALUATION ---
elif selected_module == "📊 Synthetic Quality Gates":
    st.title("Automated Synthetic Evaluation Pipeline")
    st.markdown("Pass synthetic instruction-tuning datasets through an automated **LLM-as-a-Judge** scoring gate before fine-tuning deployment.")
    
    with st.container(border=True):
        q_val = st.text_input("Generated Question", value="What is Reciprocal Rank Fusion?")
        c_val = st.text_area("Context Source", value="AegisAI utilizes Reciprocal Rank Fusion to merge dense semantic embeddings with lexical matching.")
        r_val = st.text_area("Model Response", value="Reciprocal Rank Fusion merges dense and sparse results to improve retrieval recall.")
        
        if st.button("Run Quality Gate Inspection", type="primary", use_container_width=True):
            with st.spinner("Evaluating sample against faithfulness and relevance rubrics..."):
                time.sleep(0.3)
                eval_data = st.session_state.evaluator.evaluate(q_val, c_val, r_val)
                
            st.markdown("### Inspection Scorecard")
            m1, m2, m3 = st.columns(3)
            m1.metric("Faithfulness Score", f"{eval_data['faithfulness']} / 1.0")
            m2.metric("Relevance Score", f"{eval_data['relevance']} / 1.0")
            
            if eval_data["passed"]:
                m3.success("Gate Status: **PASSED ✅**")
            else:
                m3.error("Gate Status: **REJECTED ❌**")
                
            st.info(f"**Judge Feedback:** {eval_data['feedback']}")
