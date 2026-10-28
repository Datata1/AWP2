---
name: notebook
description: Create a new Jupyter notebook in the fitting notebooks/<folder>/ following the project naming and setup conventions.
argument-hint: "<initials> <topic>"
disable-model-invocation: true
---

# New notebook

Arguments: `$ARGUMENTS` → first word = author initials (lowercase), rest = topic.
If initials are missing, derive them from `git config user.name` and confirm.

1. Pick the topic folder (report order): `01_daten`, `02_validierung`, `03_baseline`, `04_kombiniert`, `05_hierarchisch`, `06_multi-task`, `07_sequenz`, `08_datenreduktion`, `09_analyse`.
   Ask the user if the topic does not clearly fit one; never create notebooks in `archiv/`.
2. Determine the next number: highest `NN_` prefix in `notebooks/<folder>/*.ipynb` + 1, two
   digits (start at `01`).
3. File name: `notebooks/<folder>/<NN>_<initials>_<topic-kebab-case>.ipynb` (`_` between the
   parts, `-` inside the topic).
4. Create the notebook (nbformat 4, kernel `python3`, language python) with these cells:

   **Markdown:**
   ```markdown
   # <Topic in German title case>

   **Autor:** <initials> · **Datum:** <YYYY-MM-DD>

   **Ziel:** <one sentence – ask the user if unclear>
   ```

   **Code (setup):**
   ```python
   %load_ext autoreload
   %autoreload 2

   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd

   from awp2.config import FIGURES_DIR, SEED
   from awp2.data import band_columns, load_split, load_train, wavelengths
   from awp2.experiment import RunConfig, TuneConfig, run, tune
   from awp2.preprocessing import PreprocessingConfig
   from awp2.plots import plot_spectra, save_doc_figure
   ```

   **Code (data):**
   ```python
   train = load_train()
   bands = band_columns(train)
   train.shape
   ```

5. Remind the user: start with `make lab`; reusable code belongs in `src/awp2/`, not in the
   notebook; outputs are stripped on commit by nbstripout.
