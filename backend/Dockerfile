# Local Mama lead backend — Render.
#
# No audio stack, no LiveKit, no model weights. This image exists to run a few
# SQL statements, two HTTP clients and some string matching, and it should stay
# small enough that a deploy is not something you plan around.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first: cached until requirements.txt changes, so ordinary code
# deploys skip the install entirely.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --upgrade pip && pip install -r backend/requirements.txt

# The contract is shared with the agent and is imported as a top-level package
# by both. It is copied before the service so a contract change busts less.
COPY contract/ contract/
COPY backend/ backend/

RUN useradd --create-home --uid 10001 mami && chown -R mami /app
USER mami

EXPOSE 8000

# Render sets $PORT. Migrations are NOT run here — they are a pre-deploy
# command (see render.yaml), so they run once per deploy rather than once per
# container, and a rollback does not re-run them.
CMD ["sh", "-c", "uvicorn backend.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
