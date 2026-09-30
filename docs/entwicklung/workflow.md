# Arbeitsablauf

So arbeiten wir mit dem Repo – vom Issue bis zum Merge. Aufgaben und Zuständigkeiten stehen im
[Board „AWP2"](https://github.com/users/Datata1/projects/2).

## Einmalig

```bash
git clone https://github.com/Datata1/AWP2.git
cd AWP2
make setup
```

Danach die Rohdaten nach `data/raw/` und die Kursunterlagen aus ILIAS nach `data/assets/`
kopieren (siehe `data/README.md`).

Stundendoku anlegen: `/stunden init` (siehe [Stundendoku](#stundendoku)).

## Pro Aufgabe

| # | Schritt | Mit Claude | Ohne Claude | Board |
| --- | --- | --- | --- | --- |
| 1 | **Aufgabe festhalten** – jede Arbeit beginnt mit einem Issue | `/issue <Aufgabe>` | GitHub → *New issue* | Todo |
| 2 | **Übernehmen** – Issue zuweisen, eigenen Branch anlegen | `/start <nr>` | `gh issue develop <nr> --name <typ>/<nr>-<name> --checkout` | In Progress |
| 3 | **Arbeiten** – Code in `src/`, Notebooks in `notebooks/`, oft committen | `/experiment …`, `/notebook …` | normal arbeiten, `git commit` | |
| 4 | **Abgeben** – PR öffnen, eine andere Person reviewt | `/pr` | `gh pr create` (Vorlage ausfüllen) | Review |
| 5 | **Mergen** – Reviewer:in klickt *Squash and merge*, Issue schließt sich | | | Done |

!!! note "Issue bleibt nach dem Merge offen?"
    GitHub verknüpft `Closes #nr` gelegentlich nicht. Dann steht im PR rechts unter
    *Development* kein Issue. Issue dort verknüpfen oder nach dem Merge von Hand schließen –
    `/pr` prüft das automatisch, `/status` findet vergessene Fälle.
| 6 | **Aufräumen** | | `git switch main && git pull` | |

Das Board wird zum Teil automatisch gepflegt: Neue Issues landen auf *Todo*, geschlossene auf
*Done*. *In Progress* und *Review* setzen `/start` und `/pr` – ohne Claude das Issue im Board von
Hand verschieben.

### Board-Ansichten

Jedes Issue hat einen **Bereich**: Orga, Domäne, EDA, Data Prep, Modellierung, Abgabe
(`/issue` setzt ihn automatisch, sonst im Board von Hand). Das Board hat drei Tabs:

| Tab | Zeigt | Wofür |
| --- | --- | --- |
| **Aktuell** | Kanban des aktuellen Meilensteins, Spalten = Status | Tagesgeschäft, Statusmeeting |
| **Nach Bereich** | Offene Issues als Tabelle, gruppiert nach Bereich | Überblick, Planung |
| **Meine** | Nur mir zugewiesene offene Issues | Was mache ich als Nächstes? |

## Grundregeln

- **`main` ist immer lauffähig.** Niemand committet direkt auf `main`; alles läuft über Branch + PR.
- **Ein Issue = ein Branch = ein PR.** Kleine Aufgaben halten Reviews schnell und vermeiden Konflikte.
- **Aktuell bleiben:** vor einem neuen Branch `main` pullen; bei längeren Branches zwischendurch
  `git pull origin main`.
- **Blockiert?** Label `blocked` setzen und im Issue in einer Zeile schreiben, worauf gewartet wird.
- **Kurz schreiben:** Issues, PRs und Kommentare knapp und auf Deutsch – Details stehen im Code
  und in `docs/`.
- Branch-Namen, Commit-Typen und Sprache: siehe [Konventionen](konventionen.md#git-workflow).

## Review

- Jeder PR wird von einer anderen Person gelesen, bevor er gemergt wird.
- Bei Code in `src/` oder `notebooks/`: vorher den `ml-reviewer` laufen lassen (`/pr` bietet es an).
- Kommentare nur zu Fehlern, Unklarheiten oder Konventionen – Kleinigkeiten nicht blockieren.
- Wer reviewt, mergt auch.

## Wochenrhythmus

| Wann | Was |
| --- | --- |
| Täglich | Stundendoku mit `/stunden <stunden>` ([Details](#stundendoku)) |
| Vor dem Statusmeeting | `/status` – Zusammenfassung aus Board, PRs und Experimenten; spätestens 24 h vorher `make stunden` und `.xlsx` nach BWSyncAndShare |
| Im Meeting | Notizen machen |
| Nach dem Meeting | `/protokoll <Notizen>`, neue Aufgaben mit `/issue` anlegen und verteilen |
| Laufend | Board aktuell halten, Experimente in [Experimente](../modelle/experimente.md) eintragen |

## Stundendoku

Jede Person führt ihre Stundendokumentation tagesgenau (Pflicht, spätestens 24 h vor jedem
Statusmeeting aktuell). Die Datei liegt lokal in `stundendoku/` und ist **nicht in git**.

| Wann | Was |
| --- | --- |
| Einmalig | Vorlage aus ILIAS als `data/assets/Stundendokumentation_Vorlage.csv` ablegen, dann `/stunden init` |
| Am Ende jedes Arbeitstags | `/stunden 4.5` – Claude formuliert den Eintrag aus deinen Commits, PRs, Reviews, Issues und der Session; Meetings oder Lesezeit als Notiz dazuschreiben: `/stunden 6 Teammeeting EDA-Aufteilung` |
| Vergessen? | Beim Start von Claude erscheint ein Hinweis, für welche Tage mit Commits ein Eintrag fehlt; nachtragen mit `/stunden gestern 3` |
| Vor dem Statusmeeting | `make stunden` → `.xlsx` in den Teamordner in BWSyncAndShare hochladen |

Die **Stunden** gibst du selbst an – aus Commits lässt sich die Zeit nicht seriös ableiten.
Ohne Claude: `python3 tools/timesheet.py set heute 4.5 "<konkrete Tätigkeiten>"`.
