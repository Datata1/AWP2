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
- Load data with `awp2.data.load_train()`; bands via `band_columns()`.
- Split with `train_test_split(..., stratify=<Crop+Stage combination>, random_state=SEED)`
  from `awp2.config`, or `StratifiedKFold` with the same seed for cross-validation.
- Put **all** preprocessing (imputation, scaling, band selection, PCA, resampling) and the model
  into one sklearn `Pipeline` so nothing is fit on validation data.
- Handle class imbalance explicitly (e.g. `class_weight="balanced"`) and note what you did.
- Make sure predicted crop/stage combinations are valid ones seen in training.

## 3. Evaluate
- `from awp2.evaluation import evaluate, plot_confusion_matrices`
- `scores = evaluate(y_val, y_pred)` — report all returned metrics.
- `plot_confusion_matrices(y_val, y_pred, FIGURES_DIR / "<id>_confusion.png")`
- Name the most confused classes and give a hypothesis why (spectral similarity, stage overlap).

## 4. Persist
- Save the fitted pipeline: `joblib.dump(pipeline, MODELS_DIR / "<id>.joblib")`.
- Append one row to the table in `docs/modelle/experimente.md`
  (Datum, ID, Autor (git user.name), Ansatz, BAcc Crop, BAcc Stage, BAcc kombiniert,
  Macro-F1 kombiniert, Samples-F1, Notiz).
- Add findings (German, 2–5 bullets) to `docs/modelle/pflanzenart.md` and/or
  `docs/modelle/entwicklungsstadium.md`. Preprocessing decisions → `docs/daten/index.md`.

## 5. Report back
Summarize in chat: what was tried, the scores vs. the previous best, the main confusion, and a
concrete suggestion for the next experiment. Offer to run the `ml-reviewer` agent and to commit
with an `exp:` message.
