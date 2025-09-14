# backend/file_utils.py
import fitz  # PyMuPDF
import docx
import os
import re

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF using PyMuPDF."""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text.strip()

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    doc = docx.Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a TXT file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def chunk_text(text: str, max_chars: int = 800, overlap: int = 100):
    """
    Split text into overlapping chunks to improve retrieval accuracy.
    """
    words = re.split(r"\s+", text)
    chunks = []
    cur_chunk = []

    cur_len = 0
    for word in words:
        cur_chunk.append(word)
        cur_len += len(word) + 1
        if cur_len >= max_chars:
            chunks.append(" ".join(cur_chunk))
            # keep last `overlap` words for context
            cur_chunk = cur_chunk[-overlap:]
            cur_len = sum(len(w) + 1 for w in cur_chunk)

    if cur_chunk:
        chunks.append(" ".join(cur_chunk))

    return chunks
