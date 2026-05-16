import streamlit as st
from utils.embedder import create_vector_store
from utils.rag import get_answer
import os

st.set_page_config(page_title="College-DOCS AI", page_icon="𖥸")
st.title("𖥸 College-Docs AI")
st.caption("Ask anything about your college PDFs/Notes")

# Sidebar
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")

    if st.button("Process Documents"):
        if uploaded_files:
            os.makedirs("data/sample_pdfs", exist_ok=True)
            for file in uploaded_files:
                with open(f"data/sample_pdfs/{file.name}", "wb") as f:
                    f.write(file.getbuffer())
            create_vector_store()
            st.success("Documents processed..! You can now ask questions.")
        else:
            st.warning("Upload PDFs first")

# Main Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about your notes/syllabus..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_answer(prompt)
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

