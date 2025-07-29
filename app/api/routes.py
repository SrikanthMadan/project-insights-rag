from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.transcript import TranscriptResponse
from app.services.transcriber import transcribe_with_diarization
from app.models.query import QueryRequest, QueryResponse
from app.rag.rag_chain import generate_answer

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    answer = generate_answer(request.question)
    return QueryResponse(answer=answer)


@router.post("/transcribe", response_model=TranscriptResponse)
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp3", ".wav", ".mp4", ".mov")):
        raise HTTPException(status_code=400, detail="Unsupported file format.")
    
    try:
        transcript_data = await transcribe_with_diarization(file)
        return transcript_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
