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

Der Ablauf (Issue → Branch → PR → Merge) steht unter [Arbeitsablauf](workflow.md). Hier nur
die Namensregeln:

**Branches:** `<typ>/<issue-nr>-<kurzer-name>`, englisch, kebab-case, z. B. `exp/12-svm-baseline`.
Typen: `feat/`, `exp/`, `fix/`, `docs/`, `chore/`.

**Commits:** [Conventional Commits](https://www.conventionalcommits.org/), englisch, Imperativ,
max. 72 Zeichen, keine Co-Author- oder „Generated with"-Zeilen:

| Typ | Wofür | Beispiel |
| --- | --- | --- |
| `feat` | neue Funktionalität in `src/` | `feat: add vegetation index features` |
| `fix` | Bugfix | `fix: handle missing bands in loader` |
| `exp` | Experimente, Notebooks | `exp: add random forest baseline` |
| `data` | Laden / Preprocessing | `data: drop all-NaN water absorption bands` |
| `docs` | Dokumentation | `docs: describe band selection` |
| `refactor` | Umbau ohne Verhaltensänderung | `refactor: move split logic to src` |
| `chore` | Tooling, Abhängigkeiten | `chore: add xgboost` |

**Issues und PRs:** Deutsch, kurz. Titel ≤ 60 Zeichen; Issue = *Ziel* + *Fertig wenn*-Checkboxen;
PR = `Closes #nr` + 1–4 Stichpunkte. Labels: ein Typ (`feat`, `exp`, `data`, `docs`, `bug`,
`orga`), optional `crop` / `stage`, bei Wartezeit `blocked`.
