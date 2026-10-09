# awp2

Klassifikation von **Pflanzenart** und **Entwicklungsstadium** aus Satellitendaten von Feldern.

## Quickstart

```bash
# einmalig: uv installieren – https://docs.astral.sh/uv/
make setup   # Python + Pakete installieren, nbstripout aktivieren
make lab     # Jupyter Lab starten
make docs    # Doku im Browser (http://127.0.0.1:8000)
make help    # alle Befehle
```

windows:
```bash
uv sync                       # Synchronizes project dependencies by installing locked packages from uv.lock into .venv
uv run nbstripout --install   # Registers nbstripout as a Git hook to automatically strip output cells from Jupyter notebooks during commits
uv run jupyter lab            # Launches the JupyterLab interactive notebook workspace inside the uv virtual environment
uv run mkdocs serve           # Starts a local development server at http://127.0.0.1:8000 to preview documentation in real time
```

Rohdaten nach `data/raw/` kopieren (siehe [data/README.md](data/README.md)).
Ausführliche Anleitung: [docs/entwicklung/index.md](docs/entwicklung/index.md) bzw. `make docs`.

So arbeiten wir zusammen (Issue → Branch → PR): [docs/entwicklung/workflow.md](docs/entwicklung/workflow.md).
