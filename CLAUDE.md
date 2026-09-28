@AGENTS.md

## Claude Code specifics

- **Skills** (`.claude/skills/`):
  - `/experiment <idea>` — run a model experiment the standard way (split, pipeline, evaluation,
    experiment log). Use it for any new model or preprocessing variant.
  - `/notebook <initials> <topic>` — create a new notebook following the conventions.
  - `/protokoll <notes>` — turn meeting notes into a protocol in `docs/protokolle/`.
  - `/status [since]` — prepare the weekly status meeting summary (git + GitHub board).
  - `/issue <task>` — create a concise issue with label, milestone and board entry.
  - `/start <nr>` — assign an issue, set it "In Progress", create the linked branch.
  - `/pr [reviewer]` — open a concise PR for the current branch, set the issue to "Review".
- Board: `python3 .claude/scripts/board.py <nr> "<Status>" [--bereich <Bereich>]`
  (`--options` lists valid values). Prefer it over raw `gh project` calls (rate limits).
- **Subagent** `ml-reviewer`: use it to review ML code/notebooks before opening a pull request.
- **Hooks** (`.claude/settings.json`): Python files are auto-formatted with ruff after every
  edit — no need to run ruff manually afterwards. Writes to `data/raw/` are blocked.
- Edit notebooks with the notebook tool, but put reusable logic into `src/awp2/`.
- Before committing, create a branch if on `main`, and follow the commit conventions above.
