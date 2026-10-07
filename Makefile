.PHONY: install test lint format check serve rag-ingest

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check src tests

format:
	python -m ruff format src tests
	python -m ruff check src tests --fix

check: lint test

serve:
	python -m uvicorn customer_service_agent.api:app --reload --host 127.0.0.1 --port 8000

rag-ingest:
	python -m customer_service_agent.cli rag-ingest --path data/knowledge_base
