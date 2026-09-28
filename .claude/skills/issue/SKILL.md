---
name: issue
description: Create a concise GitHub issue (German) with label, milestone and project board entry via gh. Use when the user wants to track a task, idea or bug.
argument-hint: "<task in one sentence> [@assignee]"
---

# New issue: $ARGUMENTS

Write in **German**, short and concrete — no filler, no restating the obvious.

1. Check for duplicates: `gh issue list --state open --search "<keywords>"`. If one exists,
   show it and stop.
2. Draft:
   - **Title**: ≤ 60 characters, starts with a verb or noun phrase, no type prefix
     (e.g. "Random-Forest-Baseline für Crop", not "[EXP] Wir sollten mal …").
   - **Body** (follow `.github/ISSUE_TEMPLATE/aufgabe.yml`, max ~8 lines total):
     ```
     ### Ziel
     <1–2 sentences: what and why>

     ### Fertig wenn
     - [ ] <1–4 verifiable criteria>

     ### Hinweise
     <optional: links, dependencies (#nr); omit section if empty>
     ```
   - **Labels**: exactly one type (`feat`, `exp`, `data`, `docs`, `bug`, `orga`) plus
     `crop` / `stage` if specific to one target.
   - **Milestone**: the current one (`gh api repos/{owner}/{repo}/milestones --jq '.[] | "\(.title) \(.due_on)"'`
     → earliest open milestone whose due date is not past), unless the task clearly belongs later.
   - **Assignee**: only if given (`@me` for "ich"/"mir").
   - **Bereich** (board category): `Orga` (organisation, meetings, deliverables per person),
     `Domäne` (domain docs), `EDA`, `Data Prep` (cleaning/preprocessing pipeline),
     `Modellierung` (baselines, models, evaluation, feature importance), `Abgabe` (report,
     presentation, predictions, reproducibility).
3. Show title, labels, milestone and body in 5–10 lines and create it right away unless something
   is ambiguous:
   `gh issue create --title … --body … --label … --milestone … [--assignee …]`, then
   `python3 .claude/scripts/board.py <nr> Todo --bereich <Bereich>` (adds it to the board).
4. Reply with one line: `#<nr> <title> → <url>`.

Several tasks at once: create one issue per task, then list them as `#nr title` lines.
