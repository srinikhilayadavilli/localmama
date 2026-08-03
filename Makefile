PY ?= python3.11
BIN := .venv/bin

.PHONY: help setup test test-db agent backend outbox migrate smoke leads cost

help:
	@echo "Local Mama"
	@echo ""
	@echo "  make setup    Create the venv and install both sides"
	@echo "  make test     Run the test suite"
	@echo "  make test-db  ...including the pricing tests, against a real Postgres"
	@echo ""
	@echo "  make agent    The voice agent (LiveKit)"
	@echo "  make backend  The lead backend (Render), on :8000"
	@echo "  make outbox   Retry owed WhatsApp handoffs, once"
	@echo "  make migrate  Apply database migrations"
	@echo ""
	@echo "  make smoke    Black-box test a running backend (BASE=<url>)"
	@echo "  make leads    What the pipeline made of recent calls"
	@echo "  make cost     What recent calls cost, and how much of it is measured"

setup:
	$(PY) -m venv .venv
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt -r backend/requirements.txt -r requirements-dev.txt
	@test -f .env || cp .env.example .env

test:
	$(BIN)/python -m pytest -q

# The pricing arithmetic lives in SQL — an effective-dated rate card resolved by
# specificity — so a mock cannot test it. These skip without a DSN, which keeps
# `make test` hermetic; run this before touching a migration or a rate.
TEST_DB ?= localmama_test
test-db:
	@createdb $(TEST_DB) 2>/dev/null || true
	TEST_DATABASE_URL=postgresql:///$(TEST_DB) $(BIN)/python -m pytest -q

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

cost:
	$(BIN)/python scripts/cost.py $(ARGS)
