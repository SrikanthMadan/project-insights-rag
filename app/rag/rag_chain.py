# app/rag/rag_chain.py

from app.rag.vectorstore import get_chroma_vectorstore
from app.core.config import HF_TOKEN
from transformers import pipeline
from typing import List

# Load or create collection
vectorstore = get_chroma_vectorstore(persist=True)

# Load HuggingFace LLM for response generation
qa_pipeline = pipeline(
    "text-generation",
    model="tiiuae/falcon-7b-instruct",  # You can change this later
    tokenizer="tiiuae/falcon-7b-instruct",
    token=HF_TOKEN,
    device=0  # if GPU, or -1 for CPU
)

def retrieve_relevant_docs(query: str, k: int = 3) -> List[str]:
    results = vectorstore.query(query_texts=[query], n_results=k)
    documents = results.get("documents", [[]])[0]
    return documents

def generate_answer(query: str) -> str:
    context_docs = retrieve_relevant_docs(query)
    context = "\n".join(context_docs)

    prompt = f"""You are an assistant helping with project updates.
Context: {context}
Question: {query}
Answer:"""

    response = qa_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
    return response[0]['generated_text'].replace(prompt, "").strip()