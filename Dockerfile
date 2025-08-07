# Use PyTorch with CUDA (compatible with Hugging Face GPU Spaces)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Kolkata \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    HF_HOME=/tmp/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    TRANSCRIPTS_DIR=/tmp/data/transcripts \
    CHROMA_DB_DIR=/tmp/chroma_db \
    AUDIO_TEMP_DIR=/tmp/temp_audio \
    OMP_NUM_THREADS=4 \
    USE_CUDA=1

# Create required writable directories and set permissions
RUN mkdir -p /tmp/.cache/whisper \
             /tmp/huggingface/hub/.locks \
             /tmp/matplotlib \
             /tmp/data/transcripts \
             /tmp/chroma_db \
             /tmp/temp_audio \
 && chmod -R 777 /tmp

# Set working directory
WORKDIR /app

# Install system and Python dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1 wget build-essential \
 && pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir sentencepiece protobuf \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# (Optional) Preload FLAN-T5 model to avoid slow startup
RUN python -c "from transformers import pipeline; pipeline('text2text-generation', model='google/flan-t5-base')"

# Copy your app code
COPY . .

# Expose app port
EXPOSE 7860

# Start the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]