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

## Pro Aufgabe

| # | Schritt | Mit Claude | Ohne Claude | Board |
| --- | --- | --- | --- | --- |
| 1 | **Aufgabe festhalten** – jede Arbeit beginnt mit einem Issue | `/issue <Aufgabe>` | GitHub → *New issue* | Todo |
| 2 | **Übernehmen** – Issue zuweisen, eigenen Branch anlegen | `/start <nr>` | `gh issue develop <nr> --name <typ>/<nr>-<name> --checkout` | In Progress |
| 3 | **Arbeiten** – Code in `src/`, Notebooks in `notebooks/`, oft committen | `/experiment …`, `/notebook …` | normal arbeiten, `git commit` | |
| 4 | **Abgeben** – PR öffnen, eine andere Person reviewt | `/pr` | `gh pr create` (Vorlage ausfüllen) | Review |
| 5 | **Mergen** – Reviewer:in klickt *Squash and merge*, Issue schließt sich | | | Done |
| 6 | **Aufräumen** | | `git switch main && git pull` | |

Das Board wird zum Teil automatisch gepflegt: Neue Issues und PRs landen auf *Todo*, geschlossene
bzw. gemergte auf *Done*. *In Progress* und *Review* setzen `/start` und `/pr` – ohne Claude das
Issue im Board von Hand verschieben.

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
| Vor dem Statusmeeting | `/status` – Zusammenfassung aus Board, PRs und Experimenten; Stundenkontierung spätestens 24 h vorher |
| Im Meeting | Notizen machen |
| Nach dem Meeting | `/protokoll <Notizen>`, neue Aufgaben mit `/issue` anlegen und verteilen |
| Laufend | Board aktuell halten, Experimente in [Experimente](../modelle/experimente.md) eintragen |
