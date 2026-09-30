---
name: experiment
description: Run a model or preprocessing experiment the standard way for this project (stratified split, sklearn pipeline, standard evaluation, saved model, experiment log entry). Use whenever a new model, feature set or preprocessing variant is trained and evaluated.
argument-hint: "<short experiment idea, e.g. 'svm on 131 valid bands'>"
---

# Experiment: $ARGUMENTS

Follow these steps. Code and comments in English, documentation in German.

## 1. Prepare
- If on `main`, create a branch `exp/<kebab-name>` (derive the name from the idea).
- Read `docs/modelle/experimente.md` to know the current best results and avoid duplicates.
- Pick a short experiment id: `<kebab-name>` (used for files below).

## 2. Implement
- Reusable parts (preprocessing steps, feature functions, model builders) go into `src/awp2/`
  (e.g. `src/awp2/features.py`, `src/awp2/models/`). Reuse existing functions first.
- Run the experiment in a notebook (`/notebook` conventions) or a script under `notebooks/`.
- Use the shared pipeline (see `docs/daten/pipeline.md`):
  `result = awp2.experiment.run(model, name, preprocessor=build_preprocessor(...))` – it loads,
  deduplicates, uses the fixed 70/30 split and evaluates. Do not build your own split.
- For cross-validation use `awp2.data.cv_splits()`; keep everything in one sklearn `Pipeline`.
- New preprocessing steps: add a transformer to `awp2.preprocessing` and a flag to
  `build_preprocessor()` instead of preprocessing in the notebook.
- Both targets: native multi-output model, `MultiOutputClassifier`, or
  `CombinedLabelClassifier` (only valid combinations). Check `invalid_combinations` in the metrics.
- Handle class imbalance explicitly (e.g. `class_weight="balanced"`) and note what you did.

## 3. Evaluate
- `result.metrics` holds all scores from `awp2.evaluation.evaluate()` – report all of them.
- `plot_confusion_matrices(result.y_val, result.y_pred, FIGURES_DIR / "<id>_confusion.png")`
- Name the most confused classes and give a hypothesis why (spectral similarity, stage overlap).

## 4. Persist
- Save the fitted pipeline: `joblib.dump(pipeline, MODELS_DIR / "<id>.joblib")`.
- Append one row to the table in `docs/modelle/experimente.md`
  (Datum, ID, Autor (git user.name), Ansatz, BAcc Crop, BAcc Stage, BAcc kombiniert,
  Macro-F1 kombiniert, Samples-F1, Notiz).
- Add findings (German, 2–5 bullets) to the matching approach section in
  `docs/modelle/ansaetze.md` and update its row in the overview table (status, best experiment).
  If it beats the current best, update "Aktueller Stand" in `docs/modelle/index.md`.
  Preprocessing decisions → `docs/daten/index.md`.

## 5. Report back
Summarize in chat: what was tried, the scores vs. the previous best, the main confusion, and a
concrete suggestion for the next experiment. Offer to run the `ml-reviewer` agent and to commit
with an `exp:` message.
