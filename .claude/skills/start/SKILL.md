---
name: start
description: Start working on a GitHub issue - assign it, move it to "In Progress" on the board and create a linked branch.
argument-hint: "<issue number>"
disable-model-invocation: true
---

# Start issue #$ARGUMENTS

1. `gh issue view $ARGUMENTS --json number,title,labels,assignees,state,body` — stop if closed.
   If no number was given, list `gh issue list --assignee @me` and unassigned open issues of the
   current milestone and ask which one.
2. Make sure the working tree is clean (`git status --short`); if not, ask how to proceed.
3. `gh issue edit $ARGUMENTS --add-assignee @me`
4. `python3 .claude/scripts/board.py $ARGUMENTS "In Progress"`
5. Branch name: `<type>/<nr>-<short-kebab-title>` where type follows the issue's type label
   (`exp` → `exp/`, `data`/`feat` → `feat/`, `bug` → `fix/`, `docs` → `docs/`, `orga` → `chore/`),
   max ~40 characters, English, e.g. `exp/12-rf-baseline`.
   `git fetch origin main --quiet` then
   `gh issue develop $ARGUMENTS --name <branch> --base main --checkout`
6. Reply in 2–3 lines (German): branch name, the issue's "Fertig wenn" criteria, and — if it is an
   experiment — a hint to use `/experiment`.
