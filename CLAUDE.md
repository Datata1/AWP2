@AGENTS.md

## Claude Code specifics

- **Skills** (`.claude/skills/`):
  - `/experiment <idea>` — run a model experiment the standard way (split, pipeline, evaluation,
    experiment log). Use it for any new model or preprocessing variant.
  - `/notebook <initials> <topic>` — create a new notebook following the conventions.
  - `/protokoll <notes>` — turn meeting notes into a protocol in `docs/protokolle/`.
  - `/status [since]` — prepare the weekly status meeting summary.
- **Subagent** `ml-reviewer`: use it to review ML code/notebooks before opening a pull request.
- **Hooks** (`.claude/settings.json`): Python files are auto-formatted with ruff after every
  edit — no need to run ruff manually afterwards. Writes to `data/raw/` are blocked.
- Edit notebooks with the notebook tool, but put reusable logic into `src/awp2/`.
- Before committing, create a branch if on `main`, and follow the commit conventions above.
