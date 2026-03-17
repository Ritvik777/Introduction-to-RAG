"""
app.py — Streamlit UI for the Agentic RAG Pipeline
====================================================
Run with:  streamlit run app.py

This is the user interface. It imports from:
  - database.py  → add documents, get document count
  - chunker.py   → extract text from PDFs
  - graph.py     → ask questions, get graph image
"""

import streamlit as st
from database import add_documents, get_document_count
from chunker import extract_text_from_pdf
from graph import ask_question, get_graph_image


# ── Page Config ──────────────────────────────────────

st.set_page_config(page_title="Agentic RAG Pipeline", page_icon="🧠", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_added" not in st.session_state:
    st.session_state.docs_added = False


# ── Sidebar ──────────────────────────────────────────

with st.sidebar:
    st.title("🧠 Agentic RAG")
    st.caption("Self-corrective retrieval with LangGraph")

    doc_count = get_document_count()
    st.metric("Documents in Qdrant", doc_count)
    st.divider()

    # --- Add text documents ---
    st.subheader("📄 Add Text Documents")

    sample_docs = """The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 330 meters tall.
The Great Wall of China stretches over 13,000 miles and was built over many centuries starting from the 7th century BC.
Python is a programming language created by Guido van Rossum in 1991. It emphasizes code readability.
The human body contains 206 bones. The smallest bone is the stapes in the ear.
Water boils at 100 degrees Celsius (212°F) at sea level. At higher altitudes, it boils at lower temperatures.
The Amazon Rainforest produces about 20% of the world's oxygen and is home to 10% of all species on Earth.
Albert Einstein developed the theory of relativity in 1905 (special) and 1915 (general).
The speed of light in a vacuum is exactly 299,792,458 meters per second."""

    doc_text = st.text_area("One document per line:", value=sample_docs, height=200)

    if st.button("➕ Add Text to Database", use_container_width=True):
        lines = [line.strip() for line in doc_text.strip().split("\n") if line.strip()]
        if lines:
            with st.spinner("Chunking and embedding..."):
                count = add_documents(lines)
            st.success(f"Added {count} chunks!")
            st.session_state.docs_added = True
            st.rerun()
        else:
            st.warning("Enter some text first.")

    # --- Upload PDF ---
    st.subheader("📎 Upload PDF")
    pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if pdf_file and st.button("➕ Add PDF to Database", use_container_width=True):
        with st.spinner("Extracting text and embedding..."):
            text = extract_text_from_pdf(pdf_file)
            count = add_documents([text])
        st.success(f"Extracted and added {count} chunks from PDF!")
        st.session_state.docs_added = True
        st.rerun()

    st.divider()

    # --- LangGraph visualization ---
    st.subheader("🗺️ LangGraph Pipeline")
    try:
        graph_png = get_graph_image()
        st.image(graph_png)
    except Exception:
        st.code(
            "START → rewrite_query → retrieve → grade_documents\n"
            "  ├─ (relevant)     → generate → END\n"
            "  └─ (not relevant) → websearch_fallback → generate → END",
            language="text",
        )

    st.divider()
    with st.expander("📖 How does this work?"):
        st.markdown("""
**This is a self-corrective RAG pipeline** built with LangGraph.

**6 steps run for every question:**

1. **Rewrite Query** — LLM rewrites your question for better vector search
2. **Retrieve** — Qdrant finds the 4 most similar document chunks
3. **Grade Documents** — LLM checks each document: is it actually relevant?
4. **Route** — If relevant docs exist → generate. If not → web search fallback
5. **Generate** — LLM writes the answer using the best available context
6. **Check Hallucination** — LLM verifies the answer is supported by the context. If not → retry

**Why is this better than basic RAG?**
- Query rewriting catches poorly worded questions
- Relevance grading filters out false positives from vector search
- Hallucination check catches made-up facts and retries generation
- Web search fallback means the system never just says "I don't know"

**Tech Stack:**
Qdrant Cloud · Google Gemini Embeddings · OpenAI GPT-4o-mini · LangGraph · Streamlit
""")


# ── Main Chat Area ───────────────────────────────────

st.title("💬 Ask Your Documents")

if not st.session_state.docs_added and doc_count == 0:
    st.info("👈 Add some documents using the sidebar first, then start asking questions!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "trace" in msg:
            with st.expander("🔍 Pipeline Trace — what happened at each step"):
                for i, step in enumerate(msg["trace"], 1):
                    st.markdown(f"**Step {i}:**")
                    st.code(step, language="text")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running pipeline: rewrite → retrieve → grade → generate..."):
            result = ask_question(prompt)

        st.markdown(result["answer"])

        with st.expander("🔍 Pipeline Trace — what happened at each step"):
            for i, step in enumerate(result["steps"], 1):
                st.markdown(f"**Step {i}:**")
                st.code(step, language="text")

            used_web = bool(result.get("web_results"))
            source = "🌐 Web Search" if used_web else "📚 Qdrant Documents"
            st.markdown(f"**Source:** {source}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "trace": result["steps"],
    })
