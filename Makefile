PY ?= python3.11
BIN := .venv/bin

.PHONY: help setup test agent backend outbox migrate smoke leads

help:
	@echo "Local Mama"
	@echo ""
	@echo "  make setup    Create the venv and install both sides"
	@echo "  make test     Run the test suite"
	@echo ""
	@echo "  make agent    The voice agent (LiveKit)"
	@echo "  make backend  The lead backend (Render), on :8000"
	@echo "  make outbox   Retry owed WhatsApp handoffs, once"
	@echo "  make migrate  Apply database migrations"
	@echo ""
	@echo "  make smoke    Black-box test a running backend (BASE=<url>)"
	@echo "  make leads    What the pipeline made of recent calls"

setup:
	$(PY) -m venv .venv
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r agent/requirements.txt -r backend/requirements.txt -r requirements-dev.txt
	@test -f .env || cp .env.example .env

test:
	$(BIN)/python -m pytest -q

agent:
	$(BIN)/python -m agent.worker dev

backend:
	$(BIN)/uvicorn backend.api:app --reload --port 8000

outbox:
	$(BIN)/python -m backend.outbox_worker --status

migrate:
	$(BIN)/python -m backend.migrate

# Black-box: talks only over HTTP, so it tests the deployed artifact rather
# than the code on this laptop. BASE=https://… to point it at Render.
smoke:
	$(BIN)/python scripts/smoke.py $(or $(BASE),http://localhost:8000)

leads:
	$(BIN)/python scripts/inspect_lead.py $(ARGS)
