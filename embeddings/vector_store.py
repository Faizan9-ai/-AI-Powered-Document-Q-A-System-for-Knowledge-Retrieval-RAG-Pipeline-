import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# ✅ Use a consistent and robust embedding model
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# ✅ Create a new FAISS vector store from text chunks
def create_vector_store(chunks):
    if not chunks:
        raise ValueError("No document chunks found for embedding.")
    embeddings = get_embeddings()
    try:
        # Ensure that chunks are in the right format (list of Documents)
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")

# ✅ Save the FAISS index to disk
def save_vector_store(store, path="faiss_index"):
    if not os.path.exists(path):
        os.makedirs(path)
    store.save_local(path)

# ✅ Load the FAISS index from disk
def load_vector_store(path="faiss_index"):
    embeddings = get_embeddings()
    try:
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise RuntimeError(f"Error loading FAISS index: {e}")
