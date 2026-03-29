import streamlit as st
from utils.loader import load_and_split
from embeddings.vector_store import create_vector_store, save_vector_store, load_vector_store
from chains.retrieval_chain import get_qa_chain

st.set_page_config(page_title="LangChain Chatbot", layout="wide")
st.title("📚 Custom Knowledge Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

qa_chain = None

uploaded_files = st.file_uploader("Upload documents (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        chunks = load_and_split(uploaded_file.name)
        all_chunks.extend(chunks)
    try:
        vector_store = create_vector_store(all_chunks)
        save_vector_store(vector_store)
        qa_chain = get_qa_chain(vector_store)
        st.success("Documents processed and indexed!")
    except Exception as e:
        st.error(f"Embedding failed: {e}")

else:
    try:
        vector_store = load_vector_store()
        qa_chain = get_qa_chain(vector_store)
    except Exception as e:
        st.warning("Upload documents to start chatting.")
        qa_chain = None

if qa_chain:
    query = st.chat_input("Ask a question about your documents...")
    if query:
        result = qa_chain(query)
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", result["result"]))

        with st.expander("📄 Sources"):
            for doc in result["source_documents"]:
                st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                st.markdown(doc.page_content[:500] + "...")

for speaker, msg in st.session_state.chat_history:
    st.chat_message(speaker).write(msg)
