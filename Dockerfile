# Use PyTorch with CUDA (compatible with Hugging Face GPU Spaces)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Kolkata \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    HF_HOME=/tmp/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    TRANSCRIPTS_DIR=/tmp/data/transcripts \
    CHROMA_DB_DIR=/tmp/chroma_db \
    AUDIO_TEMP_DIR=/tmp/data/temp_audio \
    OMP_NUM_THREADS=4

# Create required writable directories and set permissions
RUN mkdir -p \
    /tmp/.cache/whisper \
    /tmp/huggingface \
    /tmp/matplotlib \
    /tmp/data/transcripts \
    /tmp/chroma_db \
    /tmp/data/temp_audio \
 && chmod -R 777 /tmp

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1 wget build-essential \
 && pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# (Optional) Preload Falcon model to avoid slow startup
RUN python -c "from transformers import pipeline; pipeline('text-generation', model='tiiuae/falcon-7b-instruct')"

# Copy your app code
COPY . .

# Expose app port
EXPOSE 7860

# Start the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]