# Local Mama — one image, three ways to run it.
#
#   default CMD              the LiveKit voice worker. This is what LiveKit
#                            Cloud Agents runs, and it is the only process that
#                            needs to be always-on.
#   python -m backend.app.main   the API + browser console. Render sets this as
#                            its dockerCommand, so it overrides the CMD below.
#   python deploy/run.py     both together, for a single host with one volume.

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
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

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

# No HEALTHCHECK on purpose. The right probe differs per command — /health on
# :8000 for the console, worker registration for the agent — so a single baked-in
# check is wrong for at least one of them and would report a healthy container as
# failed. Both hosts supply their own: Render via healthCheckPath, LiveKit Cloud
# by watching the worker register.

# `start` is LiveKit's production mode; `dev` adds file-watching and reload,
# which is wrong in a container.
CMD ["python", "-m", "backend.app.agent", "start"]
