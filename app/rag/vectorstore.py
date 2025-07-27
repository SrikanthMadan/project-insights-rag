import os
import chromadb
from chromadb.config import Settings

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")

client = None
collection = None

def get_chroma_vectorstore(persist=True):
    global client, collection

    if persist:
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=CHROMA_DB_DIR
        ))
    else:
        client = chromadb.Client(Settings())

    collection = client.get_or_create_collection(name="project_embeddings")
    return collection

def get_collection():
    return collection