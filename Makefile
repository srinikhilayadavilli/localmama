PY ?= python3.11
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help setup run cli demo voices agent agent-realtime agent-gemini call test test-ui clean

help:
	@echo "Local Mama — MVP"
	@echo ""
	@echo "  make setup   Create venv and install dependencies"
	@echo "  make run     Start the API + browser test console (http://127.0.0.1:8000)"
	@echo "  make cli     Talk to the agent in the terminal (no browser, speaks aloud)"
	@echo "  make demo    Scripted runs across all six languages"
	@echo "  make voices  List installed system voices per language"
	@echo "  make test    Run the Python test suite"
	@echo "  make test-ui Run the browser mic/voice state tests (needs node)"
	@echo "  make agent   Start the LiveKit voice worker (needs LiveKit keys)"
	@echo "  make call    Mint a Playground token to talk to the agent in a browser"
	@echo "  make agent-realtime  OpenAI Realtime speech-to-speech worker (EXPERIMENT, no state machine)"
	@echo "  make agent-gemini    Same worker on Gemini Live, for comparison"
	@echo "  make clean   Remove venv, caches, and generated data"

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	@test -f .env || cp .env.example .env
	@echo "\nReady. Next:  make run"

run:
	$(BIN)/python -m backend.app.main

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

call:
	$(BIN)/python -m backend.app.devcall

test:
	$(BIN)/python -m pytest -q

test-ui:
	node tests/frontend/mic_state.test.js
	node tests/frontend/mic_diagnostics.test.js
	node tests/frontend/mic_fallback.test.js

clean:
	rm -rf $(VENV) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f data/leads/*.json data/transcripts/*.json
