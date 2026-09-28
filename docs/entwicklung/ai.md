# Arbeiten mit Claude

Das Repo enthält ein gemeinsames Setup für [Claude Code](https://code.claude.com/docs). Alles
liegt in git und gilt damit für alle im Team.

## Was ist eingerichtet?

| Datei | Zweck |
| --- | --- |
| `AGENTS.md` | Projektregeln für KI-Agenten (und Menschen): Befehle, Code-Struktur, ML-Regeln, Stil, Git-Workflow. Toolneutral – funktioniert auch mit anderen Assistenten. |
| `CLAUDE.md` | Wird von Claude bei jedem Start geladen; bindet `AGENTS.md` ein und beschreibt Skills, Agent und Hooks. |
| `.claude/settings.json` | Technisch erzwungene Einstellungen: keine Co-Author-Zeile in Commits/PRs, Berechtigungen, Hooks. |
| `.claude/hooks/` | Automatische Aktionen (siehe unten). |
| `.claude/skills/` | Wiederkehrende Abläufe als `/befehl`. |
| `.claude/agents/` | Spezialisierte Subagents. |

## Skills

| Befehl | Wofür |
| --- | --- |
| `/experiment <Idee>` | Modell-Experiment nach Standard: Branch, stratifizierter Split, sklearn-Pipeline, `evaluate()`, Confusion Matrix, Modell speichern, Eintrag in [Experimente](../modelle/experimente.md). Claude nutzt den Skill auch von sich aus. |
| `/notebook <kürzel> <thema>` | Neues Notebook mit richtiger Nummer, Namen und Setup-Zellen. |
| `/protokoll <Notizen>` | Macht aus Stichpunkten ein Protokoll in `docs/protokolle/`. |
| `/status [seit]` | Zusammenfassung fürs Statusmeeting aus git und GitHub-Board: Arbeit je Person, Ergebnisse, Meilenstein-Stand, Blocker. |
| `/issue <Aufgabe> [@person]` | Knappes Issue mit Label, Meilenstein und Board-Eintrag. Mehrere Aufgaben auf einmal gehen auch. |
| `/start <nr>` | Issue übernehmen: dir zuweisen, Board auf „In Progress", verknüpften Branch anlegen. |
| `/pr [reviewer]` | PR für den aktuellen Branch (max. ~10 Zeilen, `Closes #nr`, bei Experimenten die Scores), Board auf „Review". |

Beispiel: `/experiment random forest auf allen gültigen Bändern mit class_weight=balanced`

## Typischer Ablauf mit GitHub

```text
/issue Random-Forest-Baseline für Crop und Stage @me   → #12 im Board (Todo)
/start 12                                              → Branch exp/12-rf-baseline (In Progress)
/experiment random forest baseline                     → Code, Scores, Experiment-Log
/pr                                                    → PR „Closes #12" (Review)
Merge des PR                                           → Issue zu, Board auf Done
```

Board: [GitHub Project „AWP2"](https://github.com/users/Datata1/projects/2) mit
Todo → In Progress → Review → Done. Meilensteine M1–M3 und Endabgabe sind als GitHub-Milestones
mit Fälligkeitsdatum angelegt, Labels: `feat`, `exp`, `data`, `docs`, `bug`, `orga`, `crop`,
`stage`, `blocked`.

Issues und PRs schreibt Claude auf Deutsch und bewusst knapp: Titel ≤ 60 Zeichen, Issue =
Ziel + „Fertig wenn"-Checkboxen, PR = `Closes #nr` + 1–4 Stichpunkte. Die Vorlagen in `.github/`
sorgen dafür, dass das auch im Browser so bleibt.

## Subagent `ml-reviewer`

Prüft Code und Notebooks auf Data Leakage, falsche Metriken, fehlende Seeds, Klassenungleichgewicht
und Verstöße gegen die Konventionen. Vor jedem Pull Request nutzen:
„Lass den ml-reviewer über meinen Branch schauen."

## Hooks

- **Nach jeder Änderung an einer `.py`-Datei** laufen `ruff format` und `ruff check --fix`.
  Verbleibende Lint-Fehler bekommt Claude direkt zurückgemeldet.
- **Schreibschutz für `data/raw/`**: Claude kann dort nichts ändern, löschen oder verschieben.

## Berechtigungen

- Ohne Nachfrage: `uv run`, `uv sync`, `make`, lesende git-Befehle, Branch wechseln/anlegen.
- Mit Nachfrage: `uv add`/`uv remove`, `git push`, alles andere Nicht-Lesende.
- Verboten: Schreiben in `data/raw/`, Force-Push, Push direkt auf `main`.

## Regeln ändern oder ergänzen

- **Für alle:** `AGENTS.md` anpassen (per Pull Request, damit alle es mitbekommen).
  Kurz und konkret halten – die Datei wird in jede Session geladen.
- **Nur für dich:** `CLAUDE.local.md` (persönliche Hinweise) oder
  `.claude/settings.local.json` (persönliche Berechtigungen). Beide sind in `.gitignore`.
- Nach Änderungen an Settings, Skills oder Agents Claude Code neu starten.
