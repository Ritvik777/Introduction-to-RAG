"""
database.py — Qdrant Vector Database
======================================
Handles everything related to storing and searching documents:
  - Connecting to Qdrant Cloud
  - Creating the collection (table) for our vectors
  - Adding documents (text → embedding → store)
  - Searching for similar documents

Uses: Qdrant Cloud + LangChain's QdrantVectorStore
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, EMBEDDING_SIZE
from embeddings import get_embedding_model
from chunker import chunk_text


# ── Connect to Qdrant Cloud ─────────────────────────

def get_qdrant_client():
    """Connect to Qdrant Cloud and return the client."""
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


qdrant_client = get_qdrant_client()


# ── Collection Setup ─────────────────────────────────

def setup_collection():
    """
    Make sure our collection exists in Qdrant.

    A "collection" is like a database table — it holds all our vectors.
    If the embedding size changed (e.g. we switched models), recreate it.
    """
    collections = [c.name for c in qdrant_client.get_collections().collections]

    if COLLECTION_NAME in collections:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        if info.config.params.vectors.size != EMBEDDING_SIZE:
            qdrant_client.delete_collection(COLLECTION_NAME)
            collections.remove(COLLECTION_NAME)

    if COLLECTION_NAME not in collections:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
        )


def get_vector_store():
    """Return a LangChain vector store connected to our Qdrant collection."""
    setup_collection()
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=get_embedding_model(),
    )


# ── Add Documents ────────────────────────────────────

def add_documents(texts: list[str]) -> int:
    """
    Take a list of texts, chunk them, embed them, and store in Qdrant.

    Steps:
      1. Split each text into smaller chunks (for better search)
      2. Convert each chunk into a vector using Google embeddings
      3. Store the vectors in Qdrant

    Returns: the number of chunks that were stored.
    """
    store = get_vector_store()
    all_chunks = []
    for text in texts:
        all_chunks.extend(chunk_text(text))
    store.add_texts(all_chunks)
    return len(all_chunks)


# ── Search Documents ─────────────────────────────────

def search_with_scores(query: str, top_k: int = 4) -> list[tuple[str, float]]:
    """
    Search Qdrant for documents similar to the query.

    Returns a list of (text, score) pairs, best match first.
    The score tells us how similar each document is to the query.
    """
    store = get_vector_store()
    results = store.similarity_search_with_score(query, k=top_k)
    return [(doc.page_content, score) for doc, score in results]


# ── Utility ──────────────────────────────────────────

def get_document_count() -> int:
    """Return how many document chunks are stored in Qdrant."""
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        return info.points_count
    except Exception:
        return 0
