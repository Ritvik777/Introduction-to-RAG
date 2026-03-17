"""
state.py — The RAG Pipeline State
====================================
Defines the data that flows through every node in the LangGraph.

Think of this as a "form" that gets filled in step by step:
  1. User fills in `question`
  2. rewrite_query fills in `rewritten_query`
  3. retrieve fills in `documents`
  4. grade_documents fills in `relevant_docs`
  5. (maybe) websearch_fallback fills in `web_results`
  6. generate fills in `answer`
  7. check_hallucination checks the answer (may loop back to 6)

The `steps` list is a trace log — every node adds a message
explaining what it did, so the UI can show the full pipeline.
"""

from typing import TypedDict, Annotated


def _merge_lists(a: list, b: list) -> list:
    """Merge two lists by concatenating them. Used for the `steps` field."""
    return a + b


class RAGState(TypedDict):
    question: str                              # what the user asked
    rewritten_query: str                       # improved query for vector search
    documents: list[tuple[str, float]]         # retrieved (text, score) pairs
    relevant_docs: list[str]                   # docs that passed relevance grading
    web_results: str                           # fallback web search results (if used)
    answer: str                                # the final generated answer
    retry_count: int                           # how many times we've retried generation
    is_grounded: bool                          # did the answer pass hallucination check?
    steps: Annotated[list[str], _merge_lists]  # trace log — each node appends here
