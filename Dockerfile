# Use a minimal Python 3.10 base image
FROM python:3.10-slim

# Set environment variables to avoid permission issues on Hugging Face Spaces
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV HF_HOME=/tmp/huggingface
ENV MPLCONFIGDIR=/tmp/matplotlib

# Create and set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .

# Install system dependencies (for ffmpeg, pyannote.audio, etc.)
RUN apt-get update && apt-get install -y \
    git ffmpeg libsndfile1 wget build-essential \
 && pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Optional: Preload Falcon model to avoid runtime download issues
RUN python -c "from transformers import pipeline; pipeline('text-generation', model='tiiuae/falcon-7b-instruct')"

# Copy entire application code into container
COPY . .

# Expose the port your app runs on
EXPOSE 7860

# Start the FastAPI app using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]