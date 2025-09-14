import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import psycopg2
import google.generativeai as genai

from embeddings import create_or_get_collection, embed_texts
from file_utils import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt, chunk_text

# ==========================
# FastAPI setup
# ==========================
app = FastAPI(title="DocQA Backend with Auth")

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
# Gemini LLM setup
# ==========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCMNOHwnU1uiWzaXz1y8dDNRE_y5CcsGLs")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ==========================
# PostgreSQL setup
# ==========================
DB_HOST = "localhost"
DB_PORT = 5432
DB_USER = "postgres"
DB_PASS = "root"
DB_NAME = "docqa"

# Connect or create DB
try:
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
except psycopg2.OperationalError:
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    tmp_conn = psycopg2.connect(dbname="postgres", user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
    tmp_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    tmp_cur = tmp_conn.cursor()
    tmp_cur.execute(f"CREATE DATABASE {DB_NAME}")
    tmp_cur.close()
    tmp_conn.close()
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)

cur = conn.cursor()

# Create tables if not exist
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS chats (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    query TEXT,
    response TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# ==========================
# Auth dependency
# ==========================
def get_user(username: str):
    cur.execute("SELECT id, username FROM users WHERE username=%s", (username,))
    return cur.fetchone()

# ==========================
# Models
# ==========================
class UserIn(BaseModel):
    username: str
    password: str

class QueryIn(BaseModel):
    query: str
    top_k: int = 4
    username: str

# ==========================
# Auth routes
# ==========================
@app.post("/register")
def register(user: UserIn):
    try:
        cur.execute("INSERT INTO users (username, password) VALUES (%s,%s) RETURNING id", (user.username, user.password))
        user_id = cur.fetchone()[0]
        conn.commit()
        return {"status": "ok", "user_id": user_id}
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        raise HTTPException(status_code=400, detail="Username already exists")

@app.post("/login")
def login(user: UserIn):
    cur.execute("SELECT id, username FROM users WHERE username=%s AND password=%s", (user.username, user.password))
    res = cur.fetchone()
    if not res:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"status": "ok", "user_id": res[0], "username": res[1]}

# ==========================
# Upload documents
# ==========================
@app.post("/upload")
async def upload(file: UploadFile = File(...), username: str = ""):
    user = get_user(username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid user")

    filename = file.filename
    ext = filename.split('.')[-1].lower()
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if ext in ("pdf",):
        text = extract_text_from_pdf(temp_path)
    elif ext in ("docx", "doc"):
        text = extract_text_from_docx(temp_path)
    elif ext in ("txt",):
        text = extract_text_from_txt(temp_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    chunks = chunk_text(text, max_chars=800)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text could be extracted.")

    collection = create_or_get_collection(COLLECTION_NAME)
    embs = embed_texts(chunks)
    ids = [f"{uuid.uuid4().hex}" for _ in chunks]
    metadatas = [{"source_filename": filename, "ord": i, "text_snippet": chunk[:200]} for i, chunk in enumerate(chunks)]
    collection.add(documents=chunks, embeddings=embs, ids=ids, metadatas=metadatas)

    return {"status": "ok", "filename": filename, "num_chunks": len(chunks)}




# ==========================
# Chat / query
# ==========================
@app.post("/query")
def query_endpoint(q: QueryIn):
    user = get_user(q.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid user")

    collection = create_or_get_collection(COLLECTION_NAME)
    q_emb = embed_texts([q.query])[0]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=q.top_k,
        include=["documents", "metadatas"]
    )

    retrieved_chunks = results["documents"][0]
    retrieved_metadata = results["metadatas"][0]

    context_text = "\n\n".join(
        [f"From {meta['source_filename']} (part {meta['ord']}): {doc}" 
         for doc, meta in zip(retrieved_chunks, retrieved_metadata)]
    )

    prompt = f"""
    You are an AI assistant. Use the following context to answer the question.
    If the answer is not in the context, say you don’t know.

    Query: {q.query}

    Context:
    {context_text}

    Provide a clear, concise, and well-formatted answer.
    """

    response = gemini_model.generate_content(prompt)
    answer_text = response.text if hasattr(response, "text") else str(response)

    # Save chat history
    cur.execute("INSERT INTO chats (user_id, question, answer) VALUES (%s,%s,%s)", (user[0], q.query, answer_text))
    conn.commit()

    return {
        "query": q.query,
        "answer": answer_text,
        "sources": retrieved_metadata
    }
# ==========================
# Get chat history
# ==========================
@app.get("/history")
def get_history(username: str):
    user = get_user(username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid user")
    
    # Use RealDictCursor to get dict rows (optional but cleaner)
    import psycopg2.extras
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(
        "SELECT question, answer, created_at FROM chats WHERE user_id=%s ORDER BY created_at DESC",
        (user[0],)
    )
    rows = cur.fetchall()  # each row is now a dict with keys: question, answer, created_at

    return {"history": rows}

