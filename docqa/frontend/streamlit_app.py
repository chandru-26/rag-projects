import streamlit as st
import requests
from pathlib import Path

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="DocQA Chat", layout="wide")
st.title("DocQA Chat with Login & History")

# ==========================
# Login / Register
# ==========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

auth_tab = st.tabs(["Login", "Register"])

with auth_tab[0]:
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login"):
        resp = requests.post(f"{API_BASE}/login", json={"username": username, "password": password})
        if resp.ok:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Logged in successfully!")
        else:
            st.error(resp.json().get("detail", "Login failed"))

with auth_tab[1]:
    new_user = st.text_input("Username", key="reg_user")
    new_pass = st.text_input("Password", type="password", key="reg_pass")
    if st.button("Register"):
        resp = requests.post(f"{API_BASE}/register", json={"username": new_user, "password": new_pass})
        if resp.ok:
            st.success("Registered successfully! Please login.")
        else:
            st.error(resp.json().get("detail", "Registration failed"))

# ==========================
# Upload + Chat (if logged in)
# ==========================
if st.session_state.logged_in:
    st.header(f"Welcome {st.session_state.username}")
    
    with st.expander("Upload document"):
        uploaded_file = st.file_uploader("PDF / DOCX / TXT", type=["pdf", "docx", "txt"])
        if uploaded_file:
            files_dir = Path("tmp_uploads")
            files_dir.mkdir(exist_ok=True)
            save_path = files_dir / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Uploading..."):
                files = {"file": (uploaded_file.name, open(save_path, "rb"), uploaded_file.type)}
                resp = requests.post(f"{API_BASE}/upload", files=files, params={"username": st.session_state.username})
            if resp.ok:
                data = resp.json()
                st.success(f"Indexed {data['num_chunks']} chunks from {data['filename']}")
            else:
                st.error(resp.text)

    st.markdown("---")
    st.header("Ask a question")
    # ==========================
# Show Chat History
# ==========================
if st.session_state.logged_in:
    st.markdown("---")
    st.header("Chat History")

    if st.button("Load History"):
        with st.spinner("Fetching chat history..."):
            resp = requests.get(f"{API_BASE}/history", params={"username": st.session_state.username})
        if resp.ok:
            data = resp.json()
            history = data.get("history", [])
            if history:
                for h in history:
                    st.markdown(f"**Q:** {h['question']}")
                    st.markdown(f"**A:** {h['answer']}")
                    st.markdown(f"*Timestamp:* {h['created_at']}")
                    st.markdown("---")
            else:
                st.info("No chat history found.")
        else:
            st.error(resp.text)

    query = st.text_input("Enter your question")
    top_k = st.slider("Retrieve top K chunks", 1, 8, 4)
    
    if st.button("Send"):
        if not query:
            st.warning("Type a question first.")
        else:
            with st.spinner("Fetching answer..."):
                resp = requests.post(f"{API_BASE}/query", json={
                    "query": query,
                    "top_k": top_k,
                    "username": st.session_state.username
                })
            if resp.ok:
                data = resp.json()
                st.subheader("Answer")
                st.write(data.get("answer", "No answer"))
                st.subheader("Sources / Context")
                for meta in data.get("sources", []):
                    st.write(meta)
            else:
                st.error(resp.text)
