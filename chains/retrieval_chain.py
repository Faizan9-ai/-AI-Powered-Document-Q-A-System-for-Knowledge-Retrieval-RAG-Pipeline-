from dotenv import load_dotenv
load_dotenv()

from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub, HuggingFacePipeline
from langchain.chat_models import ChatOpenAI
from transformers import pipeline
import os


def get_qa_chain(vector_store):
    """
    Create a RetrievalQA chain using either Hugging Face or OpenAI.
    """

    # === Option 1 — Use Hugging Face Model (No OpenAI key needed) ===
    # ✅ Using a text2text-generation pipeline ensures compatibility
    repo_id = "google/flan-t5-large"  # Reliable open model for Q&A

    hf_pipeline = pipeline(
        "text2text-generation",
        model=repo_id,
        max_length=512,
        temperature=0.5,
        top_p=0.95,
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # === Option 2 — Use OpenAI GPT (if you have API key) ===
    # llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)

    # === Build retriever ===
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # === Create RetrievalQA chain ===
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
