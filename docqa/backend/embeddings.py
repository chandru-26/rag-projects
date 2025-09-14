# backend/embeddings.py
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List
import os

# Load model once (global)
EMB_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMB_MODEL_NAME)
    return _model

# Chroma client (local)
def get_chroma_client():
    persist_directory = "./chroma_db"
    client = chromadb.PersistentClient(path=persist_directory)
    return client


def create_or_get_collection(name="documents"):
    client = get_chroma_client()
    try:
        collection = client.get_collection(name)
    except Exception:
        collection = client.create_collection(name)
    return collection

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    # sentence-transformers returns numpy arrays; convert to list
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embs.tolist()
