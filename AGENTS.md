# AGENTS.md

Guidelines for AI coding agents (and humans) working in this repository.

## Project

Classify agricultural **crops** and their **growth stage** from satellite hyperspectral
signatures (EO-1 Hyperion, 198 bands 427–2395 nm). University challenge (Domänenprojekt 2,
WS 2026/27), team of 3, final deadline 2026-10-31.

- Primary metric: **Balanced Accuracy**; also Macro-F1 and Samples-F1 — for crop *and* stage.
- Task, milestones, grading, deliverables: `docs/projekt/`. Data description: `docs/daten/`.
- Strong class imbalance (rice: 93 rows) and not every crop/stage combination exists.

## Commands

| Task | Command |
| --- | --- |
| Install / sync env | `make setup` (first time), `uv sync` |
| Run Python | `uv run python …` — never bare `python` / `pip` |
| Add a package | `uv add <pkg>` (dev tools: `uv add --group dev <pkg>`) |
| Lint / format | `make lint`, `make format` |
| Docs preview / build | `make docs`, `make docs-build` |
| Jupyter | `make lab` |

Always commit `pyproject.toml` and `uv.lock` together.

## Code layout

- Reusable logic (loading, preprocessing, features, models, evaluation) lives in `src/awp2/`.
  Notebooks in `notebooks/` are for exploration and reporting and **import** from `awp2`.
  If notebook code is needed twice, move it to `src/awp2/`.
- Paths and constants come from `awp2.config` — never hardcode paths, labels or seeds.
- Load raw data only via `awp2.data.load_train()` / `load_test()` (validated by the pandera
  schema in `src/awp2/data/schema.py`). Use `band_columns()` / `wavelengths()` for bands.
- `data/raw/` is **read-only**. Derived data → `data/interim/` or `data/processed/`
  (must be reproducible from raw), trained models → `models/`, figures → `reports/figures/`
  (not in git). Figures shown in the docs → `awp2.plots.save_doc_figure()` (`docs/daten/img/`).
- Plots: use and extend `awp2.plots` (e.g. `plot_spectra()`) instead of ad-hoc plotting code.
- Notebook names: `<nr>_<initials>_<topic>.ipynb`, e.g. `03_jd_baseline.ipynb`.

## ML rules

- Use `SEED` from `awp2.config` for every split, model and sampler.
- Split stratified on the crop+stage combination.
- Fit every transformation (imputer, scaler, PCA, band selection, resampling) on the training
  split only — wrap preprocessing and model in an sklearn `Pipeline`.
- Evaluate with `awp2.evaluation.evaluate()`; always report crop **and** stage metrics.
  Never report plain accuracy alone.
- Account for class imbalance (class weights, balanced sampling, appropriate metrics).
- Predictions must be valid crop/stage combinations.
- Document preprocessing decisions (e.g. dropped bands) with a reason in `docs/daten/`,
  experiment results in `docs/modelle/experimente.md`.

## Style

- **English** for identifiers, docstrings, comments and commit messages.
  **German** for everything in `docs/` and reports.
- Diagrams in docs: simple **Mermaid** (` ```mermaid `, usually `flowchart TD`, ≤ ~8 nodes,
  short labels); split or use a table when it grows.
- ruff (line length 100) — `make format` before committing.
- Every function has type hints for all arguments **and the return type** (enforced by ruff
  `ANN`), plus a short docstring if public. `make lint` type-checks `src/` with **ty** (pinned
  version – update deliberately). Return named types instead of bare tuples/sets
  (`NamedTuple`, pydantic model) so the caller sees what comes back.
- **No magic values** in code: column names, labels, sizes, thresholds, seeds, patterns and
  paths are named constants in `awp2.config` (label sets as `Literal` types there). Only purely
  local presentation details (e.g. a plot's `figsize`) may stay inline.
- Configs and results at API boundaries are frozen, strict **pydantic** models with field
  descriptions; new options become a field there, not a loose function argument. sklearn
  estimators/transformers stay plain classes (sklearn's `clone`/`get_params` conventions break
  with pydantic).
- Keep it simple: small functions, no premature abstractions.

## Comments

Code says **what** happens; comments only say **why** – a reason, constraint, trade-off or
non-obvious domain fact that the code cannot express.

- Never describe what the code does. If it seems necessary, refactor instead: clearer names,
  a well-named variable or a small extracted function.
- Never refer to the past or to changes ("previously", "now uses", "changed from", "new",
  "fixed", "instead of the old …"). Describe only the current state; history lives in git.
- No section banners, no commented-out code, no comments restating a name or type.
- A TODO needs an issue: `# TODO(#30): …`.
- Docstrings describe the contract (inputs, outputs, guarantees) of public functions, not the
  implementation steps.
- When editing a file, remove comments in the touched code that break these rules.
- Notebook markdown cells may narrate the analysis – they are documentation, not code comments.

```python
# bad: says what, refers to the past
# Transpose the frame, previously we used the column median here
filled = spectra.T.interpolate(method="index").T

# good: says why
# "_" occurs in crop and stage names, so it cannot separate them
LABEL_SEP = "|"
```

## Git workflow

- Never commit directly to `main`; branch, push, open a pull request, merge after review.
- Branch names: `feat/…`, `exp/…`, `fix/…`, `docs/…`, `chore/…` (kebab-case),
  e.g. `exp/svm-baseline`.
- [Conventional Commits](https://www.conventionalcommits.org/), imperative, English,
  subject ≤ 72 characters:

  | Type | Use for |
  | --- | --- |
  | `feat` | new functionality in `src/` |
  | `fix` | bug fix |
  | `exp` | experiments, notebooks, model runs |
  | `data` | data loading / preprocessing pipelines |
  | `docs` | documentation |
  | `refactor` | restructuring without behavior change |
  | `chore` | tooling, dependencies, config |

  Example: `exp: add random forest baseline for crop classification`
- **No AI attribution**: no `Co-Authored-By` trailers or "Generated with …" lines in commits
  or pull requests.
- Link work to issues: branch `<type>/<issue-nr>-<name>`, PR body `Closes #<nr>`.
- Notebook outputs are stripped on commit by nbstripout (installed via `make setup`).
- Never commit data files, models or secrets. The repository is **public**.

## GitHub (issues, PRs, project board)

- Use the `gh` CLI for everything on GitHub (issues, PRs, labels, milestones, project).
- Who does what is tracked in issues + the project board **AWP2** (owner `Datata1`):
  Status `Todo` → `In Progress` → `Review` → `Done`. Milestones M1–M3 + Endabgabe.
- Board field **Bereich** groups issues: `Orga`, `Domäne`, `EDA`, `Data Prep`, `Modellierung`,
  `Abgabe` — set it for every new issue.
- Labels: one type (`feat`, `exp`, `data`, `docs`, `bug`, `orga`), optionally `crop` / `stage`,
  `blocked` when waiting.
- Issues and PRs are written in **German** and kept **short**: no filler, no repeating the
  title, bullets over prose, reference `#nr` instead of re-explaining.
  - Issue: title ≤ 60 chars; body = `Ziel` (1–2 sentences), `Fertig wenn` (1–4 checkboxes),
    optional `Hinweise`.
  - PR: `Closes #nr`, `Was` (1–4 bullets), `Ergebnis` only for experiments (scores vs. best).
  - Comments: only decisions, results or blockers — one to three lines.
