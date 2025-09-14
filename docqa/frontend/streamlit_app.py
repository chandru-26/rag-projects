import streamlit as st
import requests
from pathlib import Path
import os

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="DocQA (Local)", layout="wide")

st.title("DocQA — Upload documents and ask questions (local)")

# -------------------------
# Upload Section
# -------------------------
with st.expander("Upload document"):
    uploaded_file = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        files_dir = Path("tmp_uploads")
        files_dir.mkdir(exist_ok=True)
        save_path = files_dir / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # send to backend
        with st.spinner("Uploading and indexing…"):
            files = {"file": (uploaded_file.name, open(save_path, "rb"), uploaded_file.type)}
            resp = requests.post(f"{API_BASE}/upload", files=files)
        if resp.ok:
            data = resp.json()
            st.success(f"Indexed {data['num_chunks']} chunks from {data['filename']}")
        else:
            st.error(f"Upload failed: {resp.text}")

st.markdown("---")
st.header("Ask a question")

# -------------------------
# Query Section
# -------------------------
query = st.text_input("Enter your question here")
top_k = st.slider("Retrieve top K chunks", 1, 8, 4)

if st.button("Ask"):
    if not query.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Querying backend and generating answer…"):
            resp = requests.post(f"{API_BASE}/query", json={"query": query, "top_k": top_k})
        if not resp.ok:
            st.error(f"Query failed: {resp.text}")
        else:
            res = resp.json()
            answer = res.get("answer", "No answer generated.")
            sources = res.get("sources", [])

            st.subheader("Answer")
            st.markdown(answer)

            if sources:
                st.markdown("---")
                st.subheader("Sources")
                for i, src in enumerate(sources, 1):
                    st.write(f"**{i}. {src.get('source_filename','-')}** (chunk {src.get('ord','-')})")
                    st.caption(src.get("text_snippet", "")[:300])

# -------------------------
# Sidebar Notes
# -------------------------
st.sidebar.info("""
### Notes
- Upload documents (PDF/DOCX/TXT).  
- The backend extracts text, chunks it, and stores embeddings in ChromaDB.  
- When you ask a question, it retrieves the most relevant chunks and asks Gemini to generate a clean answer.  
""")
