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
2. GitHub (via gh — source of truth for who does what):
   - Closed in period: `gh issue list --state closed --search "closed:>=<YYYY-MM-DD>" --json number,title,assignees`
   - Merged PRs: `gh pr list --state merged --search "merged:>=<YYYY-MM-DD>" --json number,title,author`
   - In progress / review: `gh project item-list <nr> --owner Datata1 --format json` (project
     "AWP2", see `gh project list --owner Datata1`) → items with Status "In Progress" / "Review".
   - Current milestone progress: `gh api repos/Datata1/AWP2/milestones --jq '.[] | "\(.title): \(.closed_issues)/\(.open_issues + .closed_issues) erledigt, fällig \(.due_on[:10])"'`
   - Blockers: `gh issue list --label blocked`; open issues without assignee in current milestone.
3. New rows in `docs/modelle/experimente.md` within the period → best scores so far.
4. Open questions: items marked TODO / "Offen" in `docs/` and open action items in the latest
   protocol in `docs/protokolle/`.

Output in **German**, in chat, **max. ~20 lines**, bullets with `#nr` references instead of
explanations:

```
## Status KW <nr> – <Datum>
### Erledigt (je Person)
### In Arbeit
### Ergebnisse (beste Scores, wichtigste Erkenntnis)
### Meilenstein <n>: x/y Issues, fällig <Datum>
### Blocker / Fragen an die Dozenten
```

Then ask whether to save it as `docs/protokolle/<YYYY-MM-DD>_status.md`.
Finish with the reminder: "Stundenkontierung spätestens 24 h vor dem Statusmeeting aktualisieren."
