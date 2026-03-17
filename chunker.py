"""
chunker.py — Text Chunking and PDF Extraction
================================================
Prepares raw text for embedding:
  - Splits long text into smaller overlapping chunks
  - Extracts text from PDF files

Why chunk? Embedding models work best on short, focused passages.
A 10-page PDF as one vector would lose detail. 500-char chunks keep it precise.
"""

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a long text into smaller overlapping pieces.

    Example (chunk_size=20, overlap=5):
      "The Eiffel Tower is in Paris France" →
      ["The Eiffel Tower is", "er is in Paris Franc", "ris France"]

    Why overlap? So we don't cut a sentence in half and lose meaning.
    The overlap creates a "window" that slides across the text.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def extract_text_from_pdf(pdf_file) -> str:
    """
    Read a PDF file and return all its text as one string.

    Works with:
      - A file path (string)
      - A Streamlit UploadedFile object
      - Any file-like object with .read()
    """
    from pypdf import PdfReader

    reader = PdfReader(pdf_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)
