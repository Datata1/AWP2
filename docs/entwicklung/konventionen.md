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

## Code

- **Keine fest verdrahteten Werte:** Spaltennamen, Labels, Größen, Schwellen, Seeds, Muster und
  Pfade stehen als Konstanten in `awp2.config`. Ausnahme: reine Darstellungsdetails wie die
  Größe eines Plots.
- **Typen überall:** Alle Argumente und der Rückgabetyp jeder Funktion sind annotiert (ruff-Regeln
  `ANN` prüfen das), `ty` prüft die Typen in `make lint`. Statt namenloser Tupel/Sets benannte
  Typen zurückgeben (`NamedTuple`, pydantic-Modell), damit sofort klar ist, was zurückkommt.
- Konfigurationen und Ergebnisse an Schnittstellen sind pydantic-Modelle (streng validiert,
  unveränderlich); sklearn-Klassen bleiben normale Klassen.

## Kommentare

Der Code zeigt, **was** passiert – Kommentare erklären nur, **warum** (Grund, Einschränkung,
Abwägung, nicht offensichtliches Domänenwissen).

- Nie beschreiben, was der Code tut. Scheint das nötig, ist der Code zu kompliziert → besser
  benennen oder eine kleine Funktion herausziehen.
- Nie auf Vergangenes verweisen („früher“, „jetzt“, „geändert von“, „neu“, „statt des alten …“).
  Für spätere Leser:innen zählt nur der aktuelle Stand; die Historie steht in git.
- Keine Abschnitts-Banner, kein auskommentierter Code, keine Kommentare, die nur den Namen
  wiederholen. TODOs nur mit Issue: `# TODO(#30): …`.
- Docstrings beschreiben die Schnittstelle (Eingaben, Rückgabe, Zusicherungen), nicht die
  Umsetzung.
- Markdown-Zellen in Notebooks dürfen die Analyse erzählen – das ist Doku, kein Code-Kommentar.

Claude hält sich über `AGENTS.md` daran, der `ml-reviewer` prüft es vor jedem PR.

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

## Doku

- Neue Seite: `.md`-Datei in den passenden Ordner unter `docs/` legen – sie erscheint automatisch
  in der Navigation. Die `index.md` eines Ordners ist dessen Übersichtsseite.
- Reihenfolge und Titel steuert die `.nav.yml` im jeweiligen Ordner; nicht aufgeführte Seiten
  landen an der Stelle von `"*"`.
- Quellen als Fußnote: `Text[^1]` und am Seitenende `[^1]: Autor (Jahr): Titel.`
- Plots für die Doku mit `awp2.plots.save_doc_figure()` speichern (siehe [EDA](../daten/eda.md)).
- Diagramme als **Mermaid** (` ```mermaid `), bewusst einfach: meist `flowchart TD`, höchstens
  ~8 Knoten, kurze Beschriftungen. Wird es größer, lieber aufteilen oder als Tabelle darstellen.
- Vorschau: `make docs` → <http://127.0.0.1:8000>
