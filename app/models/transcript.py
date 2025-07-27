from pydantic import BaseModel
from typing import List

class TranscriptSegment(BaseModel):
    speaker: str
    text: str

class TranscriptResponse(BaseModel):
    filename: str
    segments: List[TranscriptSegment]