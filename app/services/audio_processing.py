import os
import tempfile
import uuid
from pathlib import Path
from app.core.config import HF_TOKEN


import whisper # type: ignore
from moviepy.editor import VideoFileClip # type: ignore
from pyannote.audio import Pipeline

TRANSCRIPTS_DIR = Path("tmp/data/transcripts")
AUDIO_TEMP_DIR = Path("tmp/data/temp_audio")

# Load Whisper model globally
whisper_model = whisper.load_model("large")

# Load pyAnnote diarization pipeline using HF token
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)


def extract_audio(input_path: str) -> str:
    """Extract audio from video or pass through audio file."""
    AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(input_path).suffix.lower()
    output_path = AUDIO_TEMP_DIR / f"{uuid.uuid4().hex}.wav"

    if ext in [".mp3", ".wav", ".m4a"]:
        return input_path
    elif ext in [".mp4", ".mov", ".avi", ".mkv"]:
        video = VideoFileClip(input_path)
        video.audio.write_audiofile(str(output_path), verbose=False, logger=None)
        return str(output_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def transcribe(audio_path: str) -> dict:
    """Transcribe audio using Whisper."""
    return whisper_model.transcribe(audio_path)


def diarize(audio_path: str) -> list:
    """Run pyAnnote diarization and return segments."""
    diarization = diarization_pipeline(audio_path)
    segments = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return segments


def align_transcript(diarized_segments: list, transcription: dict) -> str:
    """Align diarized speakers with transcript segments."""
    words = transcription.get("segments", [])
    diarized_transcript = []
    idx = 0

    for seg in diarized_segments:
        speaker_text = f"{seg['speaker']}: "
        chunk = []

        while idx < len(words) and words[idx]["start"] < seg["end"]:
            if words[idx]["start"] >= seg["start"]:
                chunk.append(words[idx]["text"])
            idx += 1

        if chunk:
            diarized_transcript.append(speaker_text + " ".join(chunk).strip())

    return "\n".join(diarized_transcript)


def process_and_save_transcript(input_file: str) -> str:
    """Full ASR + Diarization pipeline. Saves .txt and returns path."""
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    audio_path = extract_audio(input_file)
    transcript_data = transcribe(audio_path)
    diarized_segments = diarize(audio_path)
    diarized_text = align_transcript(diarized_segments, transcript_data)

    filename = Path(input_file).stem + "_diarized.txt"
    save_path = TRANSCRIPTS_DIR / filename

    with open(save_path, "w") as f:
        f.write(diarized_text)

    return str(save_path)