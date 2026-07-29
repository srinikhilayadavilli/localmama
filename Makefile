PY ?= python3.11
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help setup cli demo voices agent agent-realtime agent-gemini latency outbox test clean

help:
	@echo "Local Mama — MVP"
	@echo ""
	@echo "  make setup   Create venv and install dependencies"
	@echo "  make cli     Talk to the agent in the terminal (no browser, speaks aloud)"
	@echo "  make demo    Scripted runs across all six languages"
	@echo "  make voices  List installed system voices per language"
	@echo "  make test    Run the Python test suite"
	@echo "  make agent   Start the LiveKit voice worker (needs LiveKit keys)"
	@echo "  make agent-realtime  OpenAI Realtime speech-to-speech worker (EXPERIMENT, no state machine)"
	@echo "  make agent-gemini    Same worker on Gemini Live, for comparison"
	@echo "  make outbox  WhatsApp handoffs still owed to callers"
	@echo "  make latency Where the caller's wait goes, from real call metrics"
	@echo "  make clean   Remove venv, caches, and generated data"

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "\nReady. Next:  make run"


cli:
	$(BIN)/python -m backend.app.cli

demo:
	$(BIN)/python -m backend.app.cli --demo

voices:
	$(BIN)/python -m backend.app.cli --list-voices

agent:
	$(BIN)/python -m backend.app.agent dev

agent-realtime:
	$(BIN)/python -m backend.app.agent_realtime dev

agent-gemini:
	REALTIME_PROVIDER=gemini $(BIN)/python -m backend.app.agent_realtime dev


latency:
	$(BIN)/python -m backend.app.latency

outbox:
	$(BIN)/python -m backend.app.outbox --status

test:
	$(BIN)/python -m pytest -q


clean:
	rm -rf $(VENV) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f data/leads/*.json data/transcripts/*.json
