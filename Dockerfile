# Local Mama — API server + LiveKit voice worker in one image.
# See deploy/run.py for why both processes share a container.

FROM python:3.11-slim

# ffmpeg/libav are what the audio stack links against; the rest is what pip
# needs to build any sdist that has no wheel for this platform.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # config.py defaults to 127.0.0.1, which inside a container answers nobody.
    HOST=0.0.0.0 \
    # Mounted volume. Leads and transcripts are files, so without this every
    # deploy silently discards every lead captured since the last one.
    DATA_DIR=/data

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes, so
# ordinary code deploys skip the slow install entirely.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --upgrade pip && pip install -r backend/requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/
COPY deploy/ deploy/

# Pull the Silero VAD / turn-detector model files into the image rather than
# fetching them on first call, when a caller is already on the line waiting.
RUN python -m backend.app.agent download-files || \
    echo "download-files unavailable; models will be fetched at startup"

RUN mkdir -p /data && useradd --create-home --uid 10001 mami && chown -R mami /app /data
USER mami

EXPOSE 8000

# Not a substitute for the platform health check — this one catches a container
# that is up but no longer serving, e.g. after the API process wedges.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/health" || exit 1

CMD ["python", "deploy/run.py"]
