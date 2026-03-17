"""
nodes.py — The 6 LangGraph Node Functions
===========================================
Each function is one "step" in the RAG pipeline.
Every node receives the current state and returns updates to it.

The 6 nodes:
  1. rewrite_query        — Rewrite the question for better search
  2. retrieve             — Search Qdrant for similar documents
  3. grade_documents      — Check if each document is actually relevant
  4. websearch_fallback   — Search the web if no relevant docs found
  5. generate             — Write the final answer
  6. check_hallucination  — Verify the answer is grounded in the context

Two routing functions:
  - route_after_grading — Decides: generate directly, or web search first?
  - route_after_check   — Decides: accept the answer, or retry generation?
"""

from state import RAGState
from llm import get_llm
from database import search_with_scores


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  NODE 1 : Rewrite the question for better search
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def rewrite_query(state: RAGState) -> dict:
    """
    Use the LLM to rewrite the user's question so it works
    better for vector similarity search.

    Example:
      "What's that tall iron thing in Paris?"
        → "Eiffel Tower Paris France landmark height"
    """
    llm = get_llm()
    prompt = f"""You are a query rewriter. Rewrite the user's question
so it will match relevant documents in a vector search.

Rules:
- Output ONLY the rewritten query, nothing else
- Keep it short and keyword-rich
- Preserve the original intent

Original question: {state["question"]}

Rewritten query:"""

    response = llm.invoke(prompt)
    rewritten = response.content.strip()

    return {
        "rewritten_query": rewritten,
        "steps": [f"Rewrote query: '{state['question']}' → '{rewritten}'"],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  NODE 2 : Retrieve documents from Qdrant
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def retrieve(state: RAGState) -> dict:
    """
    Search Qdrant for the 4 most similar document chunks.
    Uses the rewritten query (not the original question) for better results.
    Returns documents paired with their similarity scores.
    """
    query = state["rewritten_query"]
    results = search_with_scores(query, top_k=4)

    doc_lines = [f"  - (score={score:.3f}) {text[:80]}..." for text, score in results]
    step_msg = "Retrieved {} documents:\n{}".format(len(results), "\n".join(doc_lines))

    return {
        "documents": results,
        "steps": [step_msg],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  NODE 3 : Grade each document for relevance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def grade_documents(state: RAGState) -> dict:
    """
    Ask the LLM to check each retrieved document:
    "Is this document actually relevant to the question?"

    This filters out documents that are similar in wording
    but don't actually help answer the question.
    """
    llm = get_llm()
    question = state["question"]
    relevant = []
    details = []

    for text, score in state["documents"]:
        prompt = f"""You are a relevance grader. Given a user question and a document,
decide if the document is relevant to answering the question.

Reply with ONLY "yes" or "no".

Question: {question}
Document: {text}

Relevant (yes/no):"""

        response = llm.invoke(prompt)
        grade = response.content.strip().lower()

        if "yes" in grade:
            relevant.append(text)
            details.append(f"  ✅ RELEVANT (score={score:.3f}): {text[:60]}...")
        else:
            details.append(f"  ❌ NOT RELEVANT (score={score:.3f}): {text[:60]}...")

    step_msg = "Graded {} documents → {} relevant:\n{}".format(
        len(state["documents"]), len(relevant), "\n".join(details)
    )

    return {
        "relevant_docs": relevant,
        "steps": [step_msg],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROUTING : Decide what to do after grading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def route_after_grading(state: RAGState) -> str:
    """
    This is the "decision point" in our graph:

      - If we found relevant documents → go to 'generate'
      - If NO documents are relevant  → go to 'websearch_fallback'
    """
    if state["relevant_docs"]:
        return "generate"
    return "websearch_fallback"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  NODE 4 : Web search fallback
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def websearch_fallback(state: RAGState) -> dict:
    """
    When no relevant documents are found in Qdrant,
    search the web using DuckDuckGo as a backup plan.

    No API key required — DuckDuckGo search is free.
    """
    from duckduckgo_search import DDGS

    question = state["question"]
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(question, max_results=3))
        web_text = "\n\n".join(
            f"**{r['title']}**\n{r['body']}" for r in results
        )
    except Exception as e:
        web_text = f"Web search failed: {e}"

    step_msg = f"No relevant docs in database — searched the web for: '{question}'"

    return {
        "web_results": web_text,
        "steps": [step_msg],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  NODE 5 : Generate the final answer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate(state: RAGState) -> dict:
    """
    The final step: generate an answer using the best available context.

    If we have relevant documents from Qdrant → use those.
    If we fell back to web search → use web results instead.
    """
    llm = get_llm()
    question = state["question"]

    if state.get("relevant_docs"):
        context = "\n\n---\n\n".join(state["relevant_docs"])
        source = "Qdrant documents"
    elif state.get("web_results"):
        context = state["web_results"]
        source = "web search"
    else:
        context = "No information available."
        source = "none"

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the context doesn't contain enough information, say so honestly.

Context (from {source}):
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    answer = response.content

    step_msg = f"Generated answer using {source} ({len(context)} chars of context)"

    return {
        "answer": answer,
        "retry_count": state.get("retry_count", 0) + 1,
        "steps": [step_msg],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  NODE 6 : Check for hallucination
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_hallucination(state: RAGState) -> dict:
    """
    Ask the LLM: "Is this answer actually supported by the context?"

    This catches cases where the LLM makes up facts that aren't
    in the retrieved documents — a common problem called "hallucination".

    Stores the result in `is_grounded` so the router can read it.
    """
    llm = get_llm()

    context = "\n\n".join(state.get("relevant_docs", [])) or state.get("web_results", "")
    answer = state["answer"]

    prompt = f"""You are a hallucination grader. Given a context and an answer,
decide if the answer is grounded in (supported by) the context.

Reply with ONLY "grounded" or "not grounded".

Context: {context}

Answer: {answer}

Verdict (grounded/not grounded):"""

    response = llm.invoke(prompt)
    verdict = response.content.strip().lower()
    is_grounded = "grounded" in verdict and "not grounded" not in verdict

    if is_grounded:
        step_msg = "✅ Hallucination check PASSED — answer is grounded in context"
    else:
        step_msg = f"⚠️ Hallucination check FAILED — answer may not be supported (attempt {state.get('retry_count', 0)}/2)"

    return {
        "is_grounded": is_grounded,
        "steps": [step_msg],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ROUTING : Decide what to do after hallucination check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def route_after_check(state: RAGState) -> str:
    """
    After checking for hallucination:

      - If the answer is grounded        → "accept" (go to END)
      - If hallucinated + retries left    → "retry"  (loop back to generate)
      - If hallucinated + no retries left → "accept" (go to END anyway)

    Max 1 retry to avoid infinite loops.
    """
    if state.get("is_grounded", True):
        return "accept"

    if state.get("retry_count", 0) < 2:
        return "retry"

    return "accept"
