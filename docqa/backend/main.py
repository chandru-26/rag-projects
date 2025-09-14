# backend/main.py
import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai  # ✅ Gemini SDK

from embeddings import create_or_get_collection, embed_texts
from file_utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    chunk_text,
)

# ==========================
# FastAPI setup
# ==========================
app = FastAPI(title="DocQA Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

COLLECTION_NAME = "documents"

# ==========================
# Configure Gemini
# ==========================
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "AIzaSyCMNOHwnU1uiWzaXz1y8dDNRE_y5CcsGLs"  # ⚠️ replace with your real key
)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ==========================
# File upload endpoint
# ==========================
@app.post("/upload")
async def upload(file: UploadFile = File(...), metadata: str = ""):
    filename = file.filename
    ext = filename.split(".")[-1].lower()
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract text
    if ext in ("pdf",):
        text = extract_text_from_pdf(temp_path)
    elif ext in ("docx", "doc"):
        text = extract_text_from_docx(temp_path)
    elif ext in ("txt",):
        text = extract_text_from_txt(temp_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use pdf/docx/txt.")

    # Chunk text with overlap
    chunks = chunk_text(text, max_chars=500, overlap=100)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text could be extracted from the document.")

    # Create embeddings and store in Chroma
    collection = create_or_get_collection(COLLECTION_NAME)
    embs = embed_texts(chunks)
    ids = [f"{uuid.uuid4().hex}" for _ in chunks]
    metadatas = [
        {"source_filename": filename, "ord": i, "text_snippet": chunk[:200]}
        for i, chunk in enumerate(chunks)
    ]
    collection.add(documents=chunks, embeddings=embs, ids=ids, metadatas=metadatas)

    return {"status": "ok", "filename": filename, "num_chunks": len(chunks)}

# ==========================
# Query endpoint with Gemini
# ==========================
class QueryIn(BaseModel):
    query: str
    top_k: int = 4


@app.post("/query")
def query_endpoint(q: QueryIn):
    collection = create_or_get_collection(COLLECTION_NAME)

    # embed query
    q_emb = embed_texts([q.query])[0]

    # retrieve more chunks than needed
    n_results = max(q.top_k * 3, q.top_k + 5)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # assemble hits
    hits = [
        {"document": d, "metadata": m, "distance": dist}
        for d, m, dist in zip(docs, metas, dists)
    ]
    hits.sort(key=lambda x: x["distance"])  # best first

    # dedupe by (filename, ord)
    seen = set()
    deduped = []
    for h in hits:
        key = (h["metadata"].get("source_filename"), h["metadata"].get("ord"))
        if key not in seen:
            seen.add(key)
            deduped.append(h)
        if len(deduped) >= q.top_k:
            break

    # merge consecutive chunks from same file
    merged = []
    i = 0
    while i < len(deduped):
        cur = deduped[i]
        cur_meta = cur["metadata"]
        cur_text = cur["document"]
        cur_ord = cur_meta.get("ord", 0)
        src = cur_meta.get("source_filename")
        j = i + 1
        while j < len(deduped):
            nm = deduped[j]["metadata"]
            if nm.get("source_filename") == src and nm.get("ord") == cur_ord + (j - i):
                cur_text += "\n\n" + deduped[j]["document"]
                j += 1
            else:
                break
        merged.append({"source": src, "ord": cur_ord, "text": cur_text})
        i = j

    # Build context for Gemini
    context_text = "\n\n".join(
        [f"From {m['source']} (part {m['ord']}):\n{m['text']}" for m in merged]
    )

    prompt = f"""
You are a precise assistant. The user question is:
"{q.query}"

Below is the retrieved context from documents. Carefully read the context and **extract ALL distinct items** that answer the question (for example: values, types, names, etc.). 

Instructions:
- Return an enumerated list (1., 2., 3., ...) with each item on its own line.
- After each item include in parentheses the source filename and part number, e.g. (file.pdf part 3).
- If you are certain the context does not contain additional items, write a final line: "No other items found in the provided context."
- If uncertain, say "I could not find more items with high confidence."

Context:
{context_text}

Answer:
"""

    # Call Gemini
    response = gemini_model.generate_content(
        prompt,
        generation_config={"max_output_tokens": 512, "temperature": 0.0}
    )
    answer_text = response.text if hasattr(response, "text") else str(response)

    return {
        "query": q.query,
        "answer": answer_text,
        "sources": deduped,  # include distances for debugging
    }
