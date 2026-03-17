"""
embeddings.py — Google Embedding Model
========================================
Turns text into a vector of 3072 numbers.

These numbers capture the "meaning" of the text,
so similar texts have similar vectors.

Uses: Google Gemini Embedding API
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GOOGLE_API_KEY, EMBEDDING_MODEL


def get_embedding_model():
    """
    Create and return the Google embedding model.

    This model converts any text into a list of 3072 numbers (a "vector").
    Similar texts will have similar vectors — that's how search works.
    """
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
