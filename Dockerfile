# Local Mama voice agent — LiveKit.
#
# The backend has its own image (backend/Dockerfile) and deploys to Render.
# This one holds a phone call and nothing else: no database driver, no Sarvam,
# no WhatsApp, no embedding model. It used to carry all of them, which put a
# ~500MB sentence-transformers model into all eight of LiveKit's idle job
# processes at once and failed calls on memory.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Shared with the backend, and imported as a top-level package by both.
COPY contract/ contract/
COPY agent/ agent/

RUN useradd --create-home --uid 10001 mami && chown -R mami /app
USER mami

# No HEALTHCHECK: LiveKit judges the worker by whether it registers, which is
# the only signal that means anything for a process with no inbound port.
CMD ["python", "-m", "agent.worker", "start"]
