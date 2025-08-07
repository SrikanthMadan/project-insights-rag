import os
import tempfile
from pathlib import Path
from typing import List, Dict
import uuid
import shutil

import whisper
from pyannote.audio import Pipeline
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from app.core.config import HF_TOKEN
from app.models.transcript import TranscriptResponse, TranscriptSegment
from app.rag.rag_chain import embed_transcript_to_chroma

TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR", "/tmp/data/transcripts"))
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_TEMP_DIR = Path(os.getenv("AUDIO_TEMP_DIR", "/tmp/data/temp_audio"))
AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = "large"
WHISPER_DEVICE = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"

# Load models once
whisper_model = whisper.load_model(WHISPER_MODEL, device=WHISPER_DEVICE)

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token=HF_TOKEN
)

SUPPORTED_VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv"]
SUPPORTED_AUDIO_EXTS = [".mp3", ".wav", ".m4a"]


def extract_audio(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    temp_audio = open(AUDIO_TEMP_DIR / f"{uuid.uuid4().hex}.wav", "wb")

    try:
        if ext in SUPPORTED_VIDEO_EXTS:
            clip = VideoFileClip(file_path)
            clip.audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
        elif ext in SUPPORTED_AUDIO_EXTS:
            audio = AudioSegment.from_file(file_path)
            audio.export(temp_audio.name, format="wav")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        temp_audio.close()
        os.unlink(temp_audio.name)
        raise RuntimeError(f"Failed to extract audio: {e}")

    return temp_audio.name


def transcribe_with_diarization(file_path: str) -> TranscriptResponse:
    print(f"Processing: {file_path}")
    audio_path = extract_audio(file_path)

    try:
        # Transcription
        result = whisper_model.transcribe(audio_path, language="en")
        segments = result.get("segments", [])

        # Diarization
        diarization = diarization_pipeline(audio_path)
        speaker_map = {}
        speaker_counter = 1
        output_segments = []

        for turn in diarization.itertracks(yield_label=True):
            segment, _, speaker = turn
            if speaker not in speaker_map:
                speaker_map[speaker] = f"Speaker {speaker_counter}"
                speaker_counter += 1

            matching_segments = [
                seg for seg in segments
                if seg["start"] >= segment.start and seg["end"] <= segment.end
            ]

            for seg in matching_segments:
                text = seg["text"].strip()
                if text:
                    output_segments.append({
                        "speaker": speaker_map[speaker],
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": text
                    })

        # Save transcript
        filename = os.path.splitext(os.path.basename(file_path))[0]
        transcript_path = TRANSCRIPTS_DIR / f"{filename}_transcript.txt"
        with open(transcript_path, "w") as f:
            for seg in output_segments:
                f.write(f"{seg['speaker']} [{seg['start']:.2f}-{seg['end']:.2f}]: {seg['text']}\n")

        # Embed in vectorstore
        embed_transcript_to_chroma(filename, output_segments)

        return TranscriptResponse(
            filename=os.path.basename(file_path),
            segments=[TranscriptSegment(**seg) for seg in output_segments]
        )

    except Exception as e:
        raise RuntimeError(f"Transcription/diarization failed: {e}")
    finally:
        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)