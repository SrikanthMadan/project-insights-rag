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

from app.rag.rag_chain import embed_transcript_to_chroma

TRANSCRIPTS_DIR = Path("/tmp/data/transcripts")
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = "large"
WHISPER_DEVICE = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"

# Load models once
whisper_model = whisper.load_model(WHISPER_MODEL).to(WHISPER_DEVICE)

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token=HF_TOKEN
)

SUPPORTED_VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv"]
SUPPORTED_AUDIO_EXTS = [".mp3", ".wav", ".m4a"]


def extract_audio(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

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


def transcribe_with_diarization(file_path: str) -> Dict:
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
                        "text": text
                    })

        # Save transcript
        filename = os.path.splitext(os.path.basename(file_path))[0]
        transcript_path = TRANSCRIPTS_DIR / f"{filename}_transcript.txt"
        with open(transcript_path, "w") as f:
            for seg in output_segments:
                f.write(f"{seg['speaker']}: {seg['text']}\n")

        # Embed in vectorstore
        embed_transcript_to_chroma(filename, output_segments)

        return {
            "file": file_path,
            "transcript_file": str(transcript_path),
            "segments": output_segments
        }

    except Exception as e:
        raise RuntimeError(f"Transcription/diarization failed: {e}")
    finally:
        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)