import os
import chromadb
from chromadb.config import Settings

DEFAULT_CHROMA_PATH = "./chroma_db" if os.getenv("ENV", "local") == "local" else "/data/chroma_db"
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", DEFAULT_CHROMA_PATH)


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