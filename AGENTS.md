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
- ruff (line length 100) — `make format` before committing.
- Type hints and a short docstring for public functions in `src/`.
- Keep it simple: small functions, no premature abstractions.

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
