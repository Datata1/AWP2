.PHONY: help setup data lab docs docs-build lint format stunden clean

help: ## Diese Hilfe anzeigen
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Umgebung installieren + nbstripout-Git-Filter aktivieren
	uv sync
	uv run nbstripout --install

data: ## Pipeline-Artefakte erzeugen (bereinigter Datensatz + Split in data/)
	uv run python -m awp2.data.artifacts

lab: ## Jupyter Lab starten
	uv run jupyter lab

docs: ## Doku lokal mit Live-Reload servieren (http://127.0.0.1:8000)
	uv run mkdocs serve

docs-build: ## Doku statisch nach site/ bauen
	uv run mkdocs build --strict

lint: ## Code prüfen (ruff + Typen mit ty)
	uv run ruff check .
	uv run ty check src

format: ## Code formatieren (ruff)
	uv run ruff format .
	uv run ruff check --fix .

stunden: ## Stundendoku anzeigen und als xlsx exportieren (für BWSyncAndShare)
	uv run python tools/timesheet.py show
	uv run python tools/timesheet.py export

clean: ## Caches und Build-Artefakte entfernen
	rm -rf site .ruff_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
