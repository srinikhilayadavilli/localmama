# Local Mama — the LiveKit voice worker.
#
# AGENT_MODULE selects the entrypoint: `agent` is the deterministic
# state-machine pipeline, `agent_realtime` the speech-to-speech one.

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
    # Leads are written here as JSON as well as to Postgres. The container
    # filesystem is ephemeral, so Postgres is the durable copy — see
    # services/lead_store.py.
    DATA_DIR=/data

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes, so
# ordinary code deploys skip the slow install entirely.
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/ backend/

# Pull the Silero VAD / turn-detector model files into the image rather than
# fetching them on first call, when a caller is already on the line waiting.
RUN python -m backend.app.agent download-files || \
    echo "download-files unavailable; models will be fetched at startup"

# Same reasoning, learned the hard way. The knowledge-base embedding model was
# fetched from HuggingFace on the first knowledge-base lookup — which happened
# mid-conversation on a real call and cost ten seconds of dead air, long enough
# that the caller asked "ఉన్నారా కాల్లో?" ("are you on the call?"). Baking it
# into the image makes the first lookup as fast as the second.
RUN python -c "from fastembed import TextEmbedding; \
TextEmbedding(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')" \
    || echo "embedding model prefetch failed; it will be fetched at runtime"

RUN mkdir -p /data && useradd --create-home --uid 10001 mami && chown -R mami /app /data
USER mami


# No HEALTHCHECK: LiveKit Cloud judges the worker by whether it registers,
# which is the only signal that means anything for a process with no inbound
# port.

# Shell form on purpose: it expands AGENT_MODULE, which is how one image runs
# either worker. `agent` is the deterministic state-machine pipeline; set
# AGENT_MODULE=agent_realtime for the speech-to-speech experiment. Two LiveKit
# Cloud agents can then share this image and differ only by their secrets.
#
# `start` is LiveKit's production mode; `dev` adds file-watching and reload,
# which is wrong in a container.
CMD python -m backend.app.${AGENT_MODULE:-agent} start
