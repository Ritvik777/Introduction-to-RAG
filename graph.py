"""
graph.py — Assemble the LangGraph Pipeline
=============================================
This file ONLY wires the nodes together into a graph.
All node logic lives in nodes.py, all state in state.py.

The graph:

  START → rewrite_query → retrieve → grade_documents ─┬─ relevant     → generate → check_hallucination ─┬─ grounded → END
                                                       └─ not relevant → websearch_fallback ──┘           └─ not grounded → generate (retry loop)
"""

from langgraph.graph import StateGraph, START, END
from state import RAGState
from nodes import (
    rewrite_query,
    retrieve,
    grade_documents,
    route_after_grading,
    websearch_fallback,
    generate,
    check_hallucination,
    route_after_check,
)


def build_rag_graph():
    """
    Build and compile the self-corrective RAG graph.

    6 nodes, connected like this:
      1. rewrite_query        — improve the question
      2. retrieve             — search Qdrant
      3. grade_documents      — filter irrelevant docs
      4. websearch_fallback   — backup if no good docs (conditional branch)
      5. generate             — write the answer
      6. check_hallucination  — verify answer is grounded (conditional loop)
    """
    graph = StateGraph(RAGState)

    # Add the 6 nodes
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("websearch_fallback", websearch_fallback)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)

    # Wire them together
    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    # Branch: grade_documents → generate OR websearch_fallback
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "websearch_fallback": "websearch_fallback"},
    )

    graph.add_edge("websearch_fallback", "generate")
    graph.add_edge("generate", "check_hallucination")

    # Loop: check_hallucination → END or back to generate
    graph.add_conditional_edges(
        "check_hallucination",
        route_after_check,
        {"accept": END, "retry": "generate"},
    )

    return graph.compile()


# Build the graph once at import time (ready to use)
rag_graph = build_rag_graph()


# ── Public API ───────────────────────────────────────

def ask_question(question: str) -> dict:
    """
    Run the full RAG pipeline for a question.

    Returns the complete state including:
      - answer: the generated response
      - steps: trace log of what each node did
      - relevant_docs: which documents were used
      - web_results: web search results (if fallback was triggered)
    """
    initial_state = {
        "question": question,
        "rewritten_query": "",
        "documents": [],
        "relevant_docs": [],
        "web_results": "",
        "answer": "",
        "retry_count": 0,
        "is_grounded": False,
        "steps": [],
    }
    return rag_graph.invoke(initial_state)


def get_graph_image() -> bytes:
    """Return the LangGraph visualization as a PNG image."""
    return rag_graph.get_graph().draw_mermaid_png()
