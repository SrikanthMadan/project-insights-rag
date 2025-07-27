import os
import tempfile
from pathlib import Path
from typing import List, Dict

import whisper
from pyannote.audio import Pipeline
from moviepy.editor import AudioFileClip, VideoFileClip
from pydub import AudioSegment

TRANSCRIPTS_DIR = Path("app/data/transcripts")
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = "large"
WHISPER_DEVICE = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"

# Load Whisper model
whisper_model = whisper.load_model(WHISPER_MODEL).to(WHISPER_DEVICE)

# Load diarization pipeline
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization@2.1",
    use_auth_token=os.environ.get("HF_TOKEN")
)

def extract_audio(file_path: str) -> str:
    """
    Convert video or audio input into a .wav file for processing.
    Returns path to temp .wav file.
    """
    ext = file_path.split(".")[-1].lower()
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    
    if ext in ["mp4", "mov"]:
        clip = VideoFileClip(file_path)
        clip.audio.write_audiofile(temp_audio.name)
    elif ext in ["mp3", "wav"]:
        audio = AudioSegment.from_file(file_path)
        audio.export(temp_audio.name, format="wav")
    else:
        raise ValueError("Unsupported file format")

    return temp_audio.name

def transcribe_with_diarization(file_path: str) -> Dict:
    """
    Performs transcription with speaker diarization and returns structured result.
    """
    print(f"Processing: {file_path}")

    # Convert input to .wav
    audio_path = extract_audio(file_path)

    # Whisper transcription
    result = whisper_model.transcribe(audio_path, language="en")
    segments = result["segments"]

    # Diarization
    diarization = diarization_pipeline(audio_path)
    
    speaker_map = {}
    speaker_counter = 1
    output_lines = []

    for turn in diarization.itertracks(yield_label=True):
        segment, _, speaker = turn
        if speaker not in speaker_map:
            speaker_map[speaker] = f"Speaker {speaker_counter}"
            speaker_counter += 1

        matching_segments = [
            seg for seg in segments
            if seg['start'] >= segment.start and seg['end'] <= segment.end
        ]

        for seg in matching_segments:
            text = seg['text'].strip()
            if text:
                line = f"{speaker_map[speaker]}: {text}"
                output_lines.append(line)

    final_transcript = "\n".join(output_lines)

    # Save transcript
    filename = os.path.splitext(os.path.basename(file_path))[0]
    transcript_path = TRANSCRIPTS_DIR / f"{filename}_transcript.txt"
    with open(transcript_path, "w") as f:
        f.write(final_transcript)

    return {
        "file": file_path,
        "transcript_file": str(transcript_path),
        "segments": output_lines
    }