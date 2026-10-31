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

Die Rohdaten liegen bereits in `data/raw/` (siehe [data/README.md](data/README.md)).
Für Notebooks und Experimente einmalig `make data` ausführen (ohne `make`:
`uv run python -m awp2.data.artifacts`); `make predict` braucht das nicht.
Ausführliche Anleitung: [docs/entwicklung/index.md](docs/entwicklung/index.md) bzw. `make docs`.

## Vorhersage für `test.csv`

Das Endmodell wird auf allen Trainingsdaten (`data/raw/train.csv`) trainiert und sagt
`data/raw/test.csv` vorher. Das Ergebnis ist `test.csv` mit den zusätzlichen Spalten `Crop` und
`Stage`.

```bash
make predict          # offizielles Modell (combined_svm_aez_month)
make predict-automl   # AutoML-Modell (automl_final)
```

Ausgabe: `reports/predictions/<modell>.csv`. Ohne `make` (z. B. unter Windows):

```bash
uv run python tools/predict_combined_svm_aez_month.py
uv run python tools/predict_combined_svm_aez_month.py --test-file <csv> --output <csv>   # optional
```

Details: [docs/modelle/endmodell.md](docs/modelle/endmodell.md).

So arbeiten wir zusammen (Issue → Branch → PR): [docs/entwicklung/workflow.md](docs/entwicklung/workflow.md).
