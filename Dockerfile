# ============================================================
#  AgentSaham — Dockerfile
# ============================================================
FROM python:3.11-slim

# Install system dependencies (ffmpeg for audio extraction)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Default command: show help
CMD ["python", "main.py", "--help"]
