import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from app.rag.vectorstore import get_chroma_vectorstore
from app.core.config import CHROMA_DB_DIR

TRANSCRIPTS_DIR = "app/data/transcripts"  # Your input .txt files

def ingest_documents():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )

    documents = []
    for filename in os.listdir(TRANSCRIPTS_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(TRANSCRIPTS_DIR, filename), "r") as f:
                content = f.read()
                chunks = text_splitter.create_documents([content])
                documents.extend(chunks)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = get_chroma_vectorstore(embedding=embeddings)
    vectorstore.add_documents(documents)

    print(f"[INFO] Ingested {len(documents)} chunks.")


if __name__ == "__main__":
    ingest_documents()