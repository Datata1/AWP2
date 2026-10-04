---
name: experiment
description: Develop and evaluate a modelling approach the project's way – tune() with cross-validation on the training part, run() once on the validation part, both tracked in MLflow, then record the result in the experiment log and the approach's page in docs/modelle/ansaetze/. Use whenever a model, feature set or preprocessing variant is trained and evaluated.
argument-hint: "<approach and idea, e.g. 'hierarchical: RF for crop, then stage per crop'>"
---

# Experiment: $ARGUMENTS

Follow `docs/modelle/ansatz-entwickeln.md`. Code and comments in English, documentation German.

## 1. Prepare
- If on `main`, create a branch `exp/<issue-nr>-<kebab-name>` (use `/start` if there is an issue).
- Make sure the artifacts exist (`data/processed/split.csv`), otherwise run `make data`.
- Read `docs/modelle/experimente.md` (current best scores) and the approach's page in
  `docs/modelle/ansaetze/` to avoid repeating what was tried.
- Decide `approach` (slug, e.g. `baseline`, `hierarchical`, `combined`) and a run `name`.

## 2. Build the approach
- The model must predict `Crop` **and** `Stage` (natively multi-output, `MultiOutputClassifier`
  or a wrapper). Reusable code goes into `src/awp2/models/<approach>.py`; reuse what exists.
- Extra preprocessing becomes a transformer in `awp2.preprocessing` plus a field in
  `PreprocessingConfig` – never ad-hoc preprocessing in a notebook.
- Handle class imbalance: `class_weight="balanced"` or `balance_samples=True`.
- Do not load or split data yourself – `tune()` and `run()` use the shared artifacts.

## 3. Tune on the training part
```python
tuned = tune(model, TuneConfig(name="<name>_search", approach="<approach>",
                               description="<what is searched and why>",
                               param_grid={...}, preprocessing=PreprocessingConfig(...)))
```
Keep grids small and meaningful. Decide only by the CV score (`tuned.best_score`).

## 4. Evaluate once on the validation part
```python
result = run(tuned.best_model, RunConfig(name="<name>", approach="<approach>",
                                         description="<one sentence>", tuning_run=tuned.run_id,
                                         preprocessing=<same as in tune>))
```
Never tune further because of the validation score – new ideas go back to step 3.

## 5. Analyse
- Report all of `result.metrics` (`model_dump()`), and `invalid_combinations` if > 0.
- In the notebook, back every finding with a plot right below its markdown cell (see
  `notebooks/01_jd_baseline.ipynb`): `plot_metric_comparison` (vs. baseline/best so far),
  `plot_tuning_candidates`, `plot_confusion_matrices`, `plot_top_confusions`,
  `plot_class_recall` from `awp2.plots` / `awp2.evaluation`.
- Name the most confused classes and a hypothesis why (spectral similarity, stage overlap, few
  samples).
- Save the key plots with `save_doc_figure(fig, "<approach>_<what>", MODEL_DOCS_FIGURES_DIR)` and
  embed them next to the matching finding on the approach page (`![…](../img/<name>.png)`).

## 6. Record
- One row in `docs/modelle/experimente.md`: Datum, ID (= run name), Autor (git user.name),
  Ansatz, BAcc Crop, BAcc Stage, BAcc kombiniert, Macro-F1 kombiniert, Samples-F1,
  MLflow-Run (first 8 chars of `result.run_id`), Notiz.
- On the approach's page `docs/modelle/ansaetze/<approach>.md` (create it from the template in
  `ansaetze/index.md` if new): one `###` section per classifier (two-stage approaches: name the
  combination, e.g. "Random Forest → SVM") with settings, results, notebook/MLflow links and
  2–5 German bullets; update the page's summary, the row in the overview table and the cell in
  the approach × classifier matrix in `ansaetze/index.md`. New overall best → "Aktueller Stand" in
  `docs/modelle/index.md`. Preprocessing decisions → `docs/daten/`.

## 7. Report back
In chat: what was tried, CV score vs. validation score, comparison with the current best, main
confusion, one concrete next step. Offer the `ml-reviewer` agent and a commit with `exp:`.
