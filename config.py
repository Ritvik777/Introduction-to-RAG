"""
config.py — All settings live here
====================================
This is the ONLY file you need to edit to configure the app.
It loads API keys from a .env file and defines constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys (loaded from .env file) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- Model Settings ---
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_SIZE = 3072  # gemini-embedding-001 outputs 3072-dimensional vectors
LLM_MODEL = "gpt-4o-mini"

# --- Qdrant Settings ---
COLLECTION_NAME = "my_documents"

# --- Chunking Settings ---
CHUNK_SIZE = 500   # max characters per chunk
CHUNK_OVERLAP = 50  # overlap between consecutive chunks
