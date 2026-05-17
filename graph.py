"""
graph.py
--------
LangGraph Workflow Engine
Implements the graph-based control flow:

  [START]
     │
     ▼
[process_query]   ← RAG retrieval + answer generation
     │
     ▼ (conditional routing)
  ┌──┴──────────────────┐
  │                     │
[output_answer]   [escalate_to_human]
  │                     │
  └──────────┬──────────┘
             ▼
           [END]

Conditional routing criteria:
- Route to output_answer   → confident retrieval + valid LLM answer
- Route to escalate_human  → low confidence / no chunks / LLM uncertain / complex intent
"""

from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from rag_engine import RAGEngine
import logging
logging.basicConfig(level=logging.INFO)

# ── State Schema ───────────────────────────────────────────────────────────────
class GraphState(TypedDict):

    question: str
    answer: Optional[str]
    confident: bool
    reason: str
    top_score: float

    sources: List[str]
    retrieved_chunks: List

    escalated: bool
    human_response: Optional[str]

    intent: Optional[str]
    escalation_message: Optional[str]


# ── Intent Detection ───────────────────────────────────────────────────────────
def detect_intent(question: str) -> str:
    """
    Enhanced intent classifier.
    """

    q = question.lower()

    complaint_keywords = [
        "broken",
        "not working",
        "error",
        "issue",
        "problem",
        "refund",
        "angry",
        "bad",
        "terrible",
        "failed",
    ]

    faq_keywords = [
        "how",
        "what",
        "where",
        "when",
        "why",
        "can i",
        "do you",
        "explain",
    ]

    billing_keywords = [
        "payment",
        "charged",
        "refund",
        "subscription",
        "billing",
        "invoice",
    ]

    technical_keywords = [
        "login",
        "server",
        "bug",
        "crash",
        "loading",
        "error",
    ]

    complex_keywords = [
        "compare",
        "difference",
        "best option",
        "recommend",
        "legal",
        "lawsuit",
    ]

    if any(k in q for k in complex_keywords):
        return "complex"

    if any(k in q for k in billing_keywords):
        return "billing"

    if any(k in q for k in technical_keywords):
        return "technical"

    if any(k in q for k in complaint_keywords):
        return "complaint"

    if any(k in q for k in faq_keywords):
        return "faq"

    return "unknown"


# ── Node Functions ─────────────────────────────────────────────────────────────

def process_query_node(state: GraphState, rag: RAGEngine) -> GraphState:
    """
    Node 1: Process Query
    - Detects intent
    - Runs full RAG pipeline
    - Populates state with result
    """
    question = state["question"]
    intent   = detect_intent(question)

    # Complex queries always escalate — require nuanced human judgment
    if intent == "complex":
        return {
            **state,

            "intent": intent,

            "answer": None,
            "confident": False,

            "reason": "complex_query",
            "top_score": 0.0,

            "sources": [],
            "retrieved_chunks": [],

            "escalated": True,
            "human_response": None,

            "escalation_message": "Complex query requires human review.",
        }

    # Run RAG
    logging.info(f"Processing query: {question}")
    logging.info(f"Detected intent: {intent}")

    result = rag.query(question)

    return {
        **state,

        "intent": intent,

        "answer": result["answer"],
        "confident": result["confident"],

        "reason": result["reason"],
        "top_score": result["top_score"],

        "sources": result["sources"],
        "retrieved_chunks": result["retrieved_chunks"],
    }


def output_answer_node(state: GraphState) -> GraphState:
    """
    Node 2a: Output Answer
    The RAG system answered with confidence — deliver it.
    """
    return {**state, "escalated": False}


def escalate_to_human_node(state: GraphState) -> GraphState:
    """
    Node 2b: Escalate to Human (HITL)
    """

    logging.warning(
        f"Escalating query due to: {state['reason']}"
    )

    escalation_messages = {
        "low_confidence": "Low retrieval confidence.",
        "no_chunks_found": "No relevant information found.",
        "llm_uncertain": "LLM expressed uncertainty.",
        "complex_query": "Complex query requires human review.",
        "sensitive_query": "Sensitive query requires escalation.",
    }

    message = escalation_messages.get(
        state["reason"],
        "Human review required."
    )

    return {
        **state,
        "escalated": True,
        "answer": None,
        "human_response": None,
        "escalation_message": message,
    }


def route_after_processing(state: GraphState) -> str:
    """
    Conditional edge function.
    Returns the name of the next node to execute.
    """
    if (
        state["confident"]
        and state["answer"]
        and state["reason"] != "sensitive_query"
    ):
        return "output_answer"
    else:
        return "escalate_to_human"


# ── Graph Builder ──────────────────────────────────────────────────────────────

def build_graph(rag: RAGEngine) -> StateGraph:
    """
    Build and compile the LangGraph workflow.
    
    Nodes:
    - process_query      : RAG retrieval + intent detection
    - output_answer      : deliver confident answer
    - escalate_to_human  : flag for HITL

    Edges:
    - START → process_query
    - process_query → (conditional) → output_answer OR escalate_to_human
    - output_answer → END
    - escalate_to_human → END
    """
    workflow = StateGraph(GraphState)

    # Add nodes (wrap process_query to inject rag dependency)
    workflow.add_node("process_query",     lambda s: process_query_node(s, rag))
    workflow.add_node("output_answer",     output_answer_node)
    workflow.add_node("escalate_to_human", escalate_to_human_node)

    # Set entry point
    workflow.set_entry_point("process_query")

    # Conditional routing after processing
    workflow.add_conditional_edges(
        "process_query",
        route_after_processing,
        {
            "output_answer":     "output_answer",
            "escalate_to_human": "escalate_to_human",
        }
    )

    # Terminal edges
    workflow.add_edge("output_answer",     END)
    workflow.add_edge("escalate_to_human", END)

    return workflow.compile()


def run_graph(question: str, rag: RAGEngine) -> GraphState:
    """
    Run the graph for a single question.
    Returns the final state.
    """
    app = build_graph(rag)

    initial_state: GraphState = {
        "question": question,
        "answer": None,
        "confident": False,
        "reason": "",
        "top_score": 0.0,

        "sources": [],
        "retrieved_chunks": [],

        "escalated": False,
        "human_response": None,

        "intent": None,
        "escalation_message": None,
    }

    final_state = app.invoke(initial_state)
    return final_state


# ── CLI Test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ingest import ingest_pipeline, vectorstore_exists

    print("Testing graph with a sample query...")
    rag    = RAGEngine()
    result = run_graph("What is RAG?", rag)
    print(f"Escalated: {result['escalated']}")
    print(f"Answer: {result['answer']}")
    print(f"Reason: {result['reason']}")
