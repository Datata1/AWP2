# Konventionen

## Repo-Struktur

| Pfad | Inhalt |
| --- | --- |
| `data/` | Daten (nicht in git) – siehe `data/README.md` |
| `notebooks/` | Jupyter-Notebooks (Exploration, Experimente) |
| `src/awp2/` | Wiederverwendbarer Python-Code (Pipelines, Features, Modelle) |
| `models/` | Trainierte Modelldateien (nicht in git) |
| `reports/` | Erzeugte Ergebnisse zum Zeigen, z. B. Plots in `figures/` |
| `docs/` | Diese Dokumentation |

## Notebooks vs. `src/`

Wiederverwendbare Logik (Laden, Vorverarbeitung, Features) gehört nach `src/awp2/`.
Notebooks bleiben für Exploration und Auswertung und importieren den Code:

```python
%load_ext autoreload
%autoreload 2

from awp2.config import RAW_DATA_DIR
```

Dank `autoreload` sind Änderungen in `src/` ohne Kernel-Neustart verfügbar.

## Namen von Notebooks

`<nr>_<kürzel>_<thema>.ipynb`, z. B. `01_jd_exploration.ipynb`. Am einfachsten mit `/notebook`.

## Sprache

- **Englisch**: Variablen- und Funktionsnamen, Docstrings, Kommentare, Commit-Messages
- **Deutsch**: alles in `docs/`, Berichte, Präsentationen

## Git-Workflow

1. Neuen Branch von `main` anlegen: `git switch -c <typ>/<kurzer-name>`
   (`feat/`, `exp/`, `fix/`, `docs/`, `chore/`), z. B. `exp/svm-baseline`.
2. Committen nach [Conventional Commits](https://www.conventionalcommits.org/)
   (englisch, Imperativ, max. 72 Zeichen):

    | Typ | Wofür | Beispiel |
    | --- | --- | --- |
    | `feat` | neue Funktionalität in `src/` | `feat: add vegetation index features` |
    | `fix` | Bugfix | `fix: handle missing bands in loader` |
    | `exp` | Experimente, Notebooks | `exp: add random forest baseline` |
    | `data` | Laden / Preprocessing | `data: drop all-NaN water absorption bands` |
    | `docs` | Dokumentation | `docs: describe band selection` |
    | `refactor` | Umbau ohne Verhaltensänderung | `refactor: move split logic to src` |
    | `chore` | Tooling, Abhängigkeiten | `chore: add xgboost` |

3. Pushen, Pull Request auf `main` öffnen, von einer zweiten Person reviewen lassen, mergen.
4. Nie direkt auf `main` pushen. Keine Co-Author- oder „Generated with"-Zeilen.
