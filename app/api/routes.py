# app/api/routes.py

from fastapi import APIRouter
from app.models.query import QueryRequest, QueryResponse
from app.rag.rag_chain import generate_answer

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    answer = generate_answer(request.question)
    return QueryResponse(answer=answer)