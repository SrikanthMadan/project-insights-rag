from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.rag.vectorstore import get_chroma_vectorstore

app = FastAPI(title="Project Insights RAG")

# Allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Initialize ChromaDB vector store on startup
@app.on_event("startup")
async def startup_event():
    get_chroma_vectorstore(persist=True)
    print("ChromaDB initialized and loaded from disk.")

@app.get("/")
async def root():
    return {"message": "RAG backend is up and running!"}