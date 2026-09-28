---
name: status
description: Prepare the weekly status meeting summary (work per team member, experiments, milestone progress, blockers).
argument-hint: "[since, e.g. 2026-10-01 or '1 week ago']"
disable-model-invocation: true
---

# Status meeting preparation

Period: since `$ARGUMENTS` (if empty: the date of the latest file in `docs/protokolle/`,
otherwise 7 days ago).

1. Collect activity:
   - `git fetch --all --quiet` (skip silently if it fails), then
     `git log --all --since="<period>" --no-merges --pretty=format:'%an|%ad|%s' --date=short`
   - Group commits by author and summarize them as work items (not a commit list).
   - Open branches not yet merged into `main`: `git branch -a --no-merged main`.
2. New rows in `docs/modelle/experimente.md` within the period → best scores so far.
3. Current milestone from `docs/projekt/zeitplan.md`: list done and open checkboxes.
4. Open questions: items marked TODO / "Offen" in `docs/` and open action items in the latest
   protocol in `docs/protokolle/`.

Output in **German**, in chat, max. ~1 page:

```
## Status KW <nr> – <Datum>
### Erledigt (je Person)
### Ergebnisse (beste Scores, wichtigste Erkenntnis)
### Meilenstein <n>: Stand
### Nächste Schritte
### Blocker / Fragen an die Dozenten
```

Then ask whether to save it as `docs/protokolle/<YYYY-MM-DD>_status.md`.
Finish with the reminder: "Stundenkontierung spätestens 24 h vor dem Statusmeeting aktualisieren."
