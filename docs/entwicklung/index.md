# Entwicklung

## Voraussetzungen

- [uv](https://docs.astral.sh/uv/getting-started/installation/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- `make` (macOS: über die Xcode Command Line Tools, Linux: meist vorinstalliert)
- `git`

## Installation

```bash
git clone <repo-url>
cd awp2
make setup
```

`make setup` installiert die richtige Python-Version, alle Pakete (aus `uv.lock`) und aktiviert
den `nbstripout`-Git-Filter, der Notebook-Outputs beim Commit entfernt.

## Häufige Befehle

| Befehl | Zweck |
| --- | --- |
| `make lab` | Jupyter Lab starten |
| `make docs` | Doku unter <http://127.0.0.1:8000> anzeigen (Live-Reload) |
| `make lint` / `make format` | Code prüfen / formatieren |
| `make help` | Alle Befehle anzeigen |

## Pakete hinzufügen

```bash
uv add <paket>               # Laufzeit-Abhängigkeit
uv add --group dev <paket>   # nur für Entwicklung
```

`pyproject.toml` **und** `uv.lock` committen, damit alle dieselbe Umgebung haben.
