---
name: pr
description: Open a concise pull request (German) for the current branch via gh, linked to its issue, and move the issue to "Review" on the board.
argument-hint: "[reviewer]"
disable-model-invocation: true
---

# Pull request for the current branch

1. Checks — stop and report if one fails:
   - Not on `main`; working tree clean (otherwise offer to commit first, following the commit
     conventions in `AGENTS.md`).
   - `make lint` passes.
2. Context:
   - Issue number: from the branch name (`<type>/<nr>-…`) or `gh issue develop --list` links;
     if none, ask (a PR without an issue is fine for small chores).
   - Changes: `git log --oneline main..HEAD` and `git diff --stat main...HEAD`.
   - If it is an experiment: the new row(s) in `docs/modelle/experimente.md`.
3. Offer to run the `ml-reviewer` agent first if `src/` or `notebooks/` changed and it has not
   been run on this branch yet.
4. Push: `git push -u origin HEAD`.
5. Create the PR — **German, max ~10 lines**, following `.github/pull_request_template.md`:
   - **Title**: same style as the commit subject (Conventional Commit, English is fine since it
     becomes the merge commit), e.g. `exp: add random forest baseline`.
   - **Body**: `Closes #<nr>` · `## Was` 1–4 bullets (what changed, not how) · `## Ergebnis`
     only for experiments: `BAcc Crop x.xx / Stage x.xx / kombiniert x.xx (bisher x.xx)` ·
     `## Review-Hinweis` only if there is something specific to check.
   - `gh pr create --base main --title … --body … [--reviewer <arg>] --assignee @me`
6. Make sure the PR body contains `Closes #<issue-nr>` literally. The repo squash-merges with the
   PR body as commit message, so this line in the commit on `main` closes the issue reliably –
   even when GitHub does not show the link in the PR sidebar (it sometimes skips parsing it:
   `gh pr view <pr> --json closingIssuesReferences` stays empty). No further action needed.
7. `python3 .claude/scripts/board.py <issue-nr> Review` (if there is an issue).
8. Reply with one line: `PR #<nr> → <url>`.
