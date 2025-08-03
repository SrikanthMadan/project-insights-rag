import os
import chromadb
from chromadb.config import Settings

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR")

client = None
collection = None

def get_chroma_vectorstore(persist=True):
    global client, collection

    if persist:
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    else:
        client = chromadb.EphemeralClient()

    collection = client.get_or_create_collection(name="project_embeddings")
    return collection

def get_collection():
    return collection