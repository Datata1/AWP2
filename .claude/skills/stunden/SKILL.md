---
name: stunden
description: Fill in the personal time sheet (Stundendokumentation) for a day - drafts a concrete German entry from the user's own git, GitHub and session activity; the user provides the hours.
argument-hint: "[init | datum] [stunden] [notizen, z. B. 'Teammeeting 1h']"
disable-model-invocation: true
---

# Stundendoku: $ARGUMENTS

Tool: `python3 tools/timesheet.py` (file in gitignored `stundendoku/` – never commit or print it
into issues/PRs; the repo is public).

## `init`
If the first argument is `init`: ask for "Nachname Vorname" if not given, run
`python3 tools/timesheet.py init "<Nachname Vorname>"`. If the template is missing, tell the user
to put `Stundendokumentation_Vorlage.csv` from ILIAS into `data/assets/`. Stop.

## Entry for a day
1. **Date**: first argument if it is a date (`YYYY-MM-DD`, `DD.MM.YYYY`, `heute`, `gestern`),
   otherwise today. If several days are missing (`python3 tools/timesheet.py missing`), offer to
   do them one after another.
2. **Collect own activity for that day** (only the user's own work):
   - `git log --all --author="$(git config user.email)" --since="<date> 00:00" --until="<date> 23:59" --format='%ad %s' --date=format:%H:%M`
   - GitHub (login via `gh api user --jq .login`):
     `gh search prs --repo Datata1/AWP2 --author @me --created <date>`,
     `gh search prs --repo Datata1/AWP2 --reviewed-by @me --updated <date>`,
     `gh search issues --repo Datata1/AWP2 --involves @me --updated <date>`
   - Meeting protocols `docs/protokolle/<date>_*.md`.
   - Work done in **this Claude session** on that day (you know it).
   - Notes from the arguments (meetings, reading, research – things git cannot see).
3. **Draft the entry** in German, concrete like the template examples: activities separated by
   `; `, name the topic (e.g. "EDA Datenqualität: fehlende Bänder und Duplikate analysiert (#15)"),
   max ~250 characters. Never vague ("Einarbeitung", "Teambesprechung" without content).
4. **Hours**: take them from the arguments. Otherwise ask – you may show first/last commit time as
   a hint, but never invent hours. Decimal with dot, e.g. `3.5`.
5. Show the draft in 2 lines (date · hours · text) and write it:
   `python3 tools/timesheet.py set <date> <hours> "<text>"` — if an entry already exists, ask
   whether to append (`--append`, adds hours and text) or keep it.
6. Reply with one line: the entry and the new total. Before a status meeting remind:
   `make stunden` → upload the `.xlsx` to the team folder in BWSyncAndShare.
