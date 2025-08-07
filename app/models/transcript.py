from pydantic import BaseModel
from typing import List

class TranscriptSegment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str

class TranscriptResponse(BaseModel):
    filename: str
    segments: List[TranscriptSegment]