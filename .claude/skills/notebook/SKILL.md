---
name: notebook
description: Create a new Jupyter notebook in notebooks/ following the project naming and setup conventions.
argument-hint: "<initials> <topic>"
disable-model-invocation: true
---

# New notebook

Arguments: `$ARGUMENTS` → first word = author initials (lowercase), rest = topic.
If initials are missing, derive them from `git config user.name` and confirm.

1. Determine the next number: highest `NN_` prefix in `notebooks/*.ipynb` + 1, two digits
   (start at `01`).
2. File name: `notebooks/<NN>_<initials>_<topic_snake_case>.ipynb`.
3. Create the notebook (nbformat 4, kernel `python3`, language python) with these cells:

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
   from awp2.data import band_columns, load_train, wavelengths
   ```

   **Code (data):**
   ```python
   train = load_train()
   bands = band_columns(train)
   train.shape
   ```

4. Remind the user: start with `make lab`; reusable code belongs in `src/awp2/`, not in the
   notebook; outputs are stripped on commit by nbstripout.
