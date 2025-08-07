from app.rag.vectorstore import get_chroma_vectorstore
from app.core.config import HF_TOKEN
from transformers import pipeline
from typing import List, Dict
import torch
from functools import lru_cache

# Load or create collection
vectorstore = get_chroma_vectorstore(persist=True)

# Lazy-load HuggingFace LLM with safe device detection
@lru_cache(maxsize=1)
def get_pipeline():
    device = 0
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        token=HF_TOKEN,
        device=device
    )

# Simple NSFW keyword filter — extendable with classifier later
NSFW_KEYWORDS = {
    "nude", "sex", "porn", "violence", "erotic", "nsfw",
    "explicit", "rape", "drugs", "weapon", "murder"
}

def is_safe(text: str) -> bool:
    text_lower = text.lower()
    return not any(word in text_lower for word in NSFW_KEYWORDS)

def retrieve_relevant_docs(query: str, k: int = 5) -> List[str]:
    results = vectorstore.query(query_texts=[query], n_results=k)
    documents = results.get("documents", [[]])[0]
    return documents

def generate_answer(query: str) -> str:
    context_docs = retrieve_relevant_docs(query)
    context = "\n".join(context_docs)

    prompt = f"""nswer the question using the following context.
Context: {context}
Question: {query}
Answer:"""

    response = get_pipeline()(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
    generated = response[0]['generated_text'].replace(prompt, "").strip()

    if not is_safe(generated):
        return "Sorry, the generated content was flagged as potentially inappropriate and has been filtered."

    return generated

def embed_transcript_to_chroma(filename: str, segments: List[Dict[str, str]]):
    """
    Converts transcript segments to a text blob and stores it in the vectorstore.
    """
    content = "\n".join(f"{seg['speaker']}: {seg['text']}" for seg in segments)
    if not content.strip():
        print(f"[WARN] Transcript for {filename} is empty. Skipping embedding.")
        return

    print(f"[INFO] Embedding transcript for: {filename}")
    vectorstore.add_texts([content], metadatas=[{"source": filename}])