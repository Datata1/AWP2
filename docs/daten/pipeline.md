# Pipeline nutzen

Die Pipeline liefert allen Ansätzen dieselben bereinigten Daten, denselben Split, dieselbe
Vorverarbeitung und dieselben Metriken. Ein Modell enthält sie bewusst nicht – das baut jeder
[Ansatz](../modelle/ansaetze/index.md) selbst. Warum sie so gebaut ist: [Warum so?](pipeline-entscheidungen.md) ·
alle Funktionen im Detail: [API-Referenz](../referenz/data.md).

```mermaid
flowchart TD
    R[data/raw] -->|make data| D[data/interim + data/processed]
    D -->|load_split| A[Ansatz: Vorverarbeitung + Modell]
    A -->|evaluate| M[Metrics]
```

## 1. Artefakte erzeugen

```bash
make data   # nach dem Kopieren der Rohdaten und wenn sich die Pipeline ändert (80/20)
# Für ein anderes Protokoll, z. B. die früheren 70/30:
make data DATA_ARGS="--validation-size 0.3"
```

| Datei | Inhalt |
| --- | --- |
| `data/interim/train_clean.parquet` | Validierte Trainingsdaten ohne Duplikate |
| `data/processed/split.csv` | Je `id`: `subset` (`train`/`val`), `cv_fold` (0–4) und Fingerabdruck der Split-Konfiguration |
| `data/processed/split_config.json` | Validierter Anteil der Validierungsdaten; sein Fingerabdruck wird beim Laden gegen `split.csv` geprüft |

Die Dateien sind nicht in git, aber deterministisch – bei allen byte-identisch.

Der Standard ist 80/20 (bis 2026-10-20: 70/30). Ein anderer Split ist ein anderes
Evaluationsprotokoll: Ergebnisse dürfen nur mit Läufen verglichen werden, die dieselbe
`split.validation_size` in MLflow haben.

## 2. Einen Ansatz aufsetzen

```python
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from awp2.data import load_split, valid_combinations
from awp2.evaluation import as_target_frame, evaluate
from awp2.preprocessing import PreprocessingConfig, build_preprocessor

split = load_split()
model = Pipeline(
    [
        ("preprocess", build_preprocessor(PreprocessingConfig(scale=True))),
        ("model", DummyClassifier(strategy="most_frequent")),  # ← euer Ansatz
    ]
)
model.fit(split.X_train, split.y_train)
y_pred = as_target_frame(model.predict(split.X_val), split.y_val.index)

metrics = evaluate(split.y_val, y_pred, valid_combinations(split.y_train))
metrics.bacc_combined
```

| Kommt aus der Pipeline | Baut jeder Ansatz selbst |
| --- | --- |
| Daten und Split: [`load_split`][awp2.data.artifacts.load_split] | Das Modell und wie es Crop **und** Stage vorhersagt |
| Vorverarbeitung: [`build_preprocessor`][awp2.preprocessing.build_preprocessor] | Zusätzliche Schritte → neues Feld in [`PreprocessingConfig`][awp2.preprocessing.PreprocessingConfig] |
| Bewertung: [`evaluate`][awp2.evaluation.evaluate] → [`Metrics`][awp2.evaluation.Metrics] | Tuning, Begründung in [Ansätze](../modelle/ansaetze/index.md), Zeile im [Experiment-Log](../modelle/experimente.md) |

Die lineare Interpolation einzelner Bandlücken ist standardmäßig auf Stützbänder mit höchstens
15 nm Abstand begrenzt. Bei größeren Abständen wird der nähere Messwert übernommen; die Grenze
lässt sich über `PreprocessingConfig(max_interpolation_gap_nm=...)` anpassen.

`AEZ` wird one-hot-kodiert. Standardmäßig ersetzt die zyklische Kodierung den numerischen
`Month` durch `Month_sin` und `Month_cos` mit einer Jahresperiode von 12 Monaten. Bei
`scale=True` werden diese beiden Merkmale mit dem Trainingsfold skaliert. Für den Vergleich mit
der numerischen Monatszahl kann `PreprocessingConfig(use_cyclic_month=False)` gesetzt werden.

### Ablauf der Vorverarbeitung

Datensicht: durchgezogen = immer, gestrichelt = optional (entfällt wenn deaktiviert),
Rauten = Verzweigung (Skalierung: genau eine Option, Reduktion: höchstens eine, Monat:
je nach `use_cyclic_month`). Jedes Kästchen nennt seine `PreprocessingConfig`-Felder.
Grafik generiert (`pipeline_options_diagram`) – bei Pipeline-Änderungen neu erzeugen,
`tools/check_pipeline_diagram.py` (CI) prüft Stand und Vollständigkeit:

```bash
uv run python -c "from awp2.preprocessing import pipeline_options_diagram; print(pipeline_options_diagram())"
```

<!-- pipeline-options-diagram:start -->
```mermaid
flowchart TD
    X["Rohdaten (Meta + Bänder)"]
    subgraph meta
        n_aez_aez["AEZ one-hot [use_meta]"]
        d_month{"Monat [use_meta]"}
        n_month_cyclic["Monat sin/cos [use_cyclic_month]"]
        n_month_month["Monat numerisch [use_cyclic_month = False]"]
        n_aez_month_aez_month["AEZ × Monat one-hot [use_aez_month]"]
    end
    subgraph albedo
        n_albedo_drop_empty["Leere Bänder entfernen (immer)"]
        n_albedo_interpolate["Lücken interpolieren (immer, max_interpolation_gap_nm)"]
        n_albedo_albedo["Mean Albedo [use_mean_albedo]"]
        n_albedo_standard_scale["Standard (wie Spektren)"]
        n_albedo_minmax_scale["Min-Max (wie Spektren)"]
        n_albedo_centering["Zentrierung (wie Spektren)"]
        n_albedo_no_scaler["keine Skalierung (wie Spektren)"]
    end
    subgraph indizes
        n_indices_drop_empty["Leere Bänder entfernen (immer)"]
        n_indices_interpolate["Lücken interpolieren (immer, max_interpolation_gap_nm)"]
        n_indices_indices["Vegetationsindizes [use_vegetation_indices]"]
        n_indices_select["Nur Indizes behalten"]
        n_indices_impute["Fehlende Indizes: Median (Training)"]
        n_indices_standard_scale["Standard [use_spectral_standard_scale]"]
        n_indices_minmax_scale["Min-Max [use_spectral_minmax_scale]"]
        n_indices_centering["Zentrierung [use_spectral_centering]"]
        n_indices_no_scaler["keine Skalierung [Standard]"]
    end
    subgraph hauptkomponenten
        n_principal_components_drop_empty["Leere Bänder entfernen (immer)"]
        n_principal_components_interpolate["Lücken interpolieren (immer, max_interpolation_gap_nm)"]
        n_principal_components_standard_scale["Standard (immer)"]
        n_principal_components_pca["PC_1 + PC_2 [use_principal_components]"]
    end
    subgraph spektren
        n_spectra_drop_empty["Leere Bänder entfernen (immer)"]
        n_spectra_interpolate["Lücken interpolieren (immer, max_interpolation_gap_nm)"]
        n_spectra_savgol["Savitzky-Golay [use_savgol + savgol_window_length + savgol_polyorder]"]
        n_spectra_snv["SNV [use_spectral_snv]"]
    end
    subgraph skalierung
        d_scaler{"Skalierung (XOR)"}
        n_spectra_standard_scale["Standard [use_spectral_standard_scale]"]
        n_spectra_minmax_scale["Min-Max [use_spectral_minmax_scale]"]
        n_spectra_centering["Zentrierung [use_spectral_centering]"]
        n_spectra_no_scaler["keine Skalierung [Standard]"]
    end
    subgraph reduktion
        d_reduction{"Reduktion (max. eine)"}
        n_spectra_pca["PCA [spectral_pca_components | braucht Standard/Center]"]
        n_spectra_ica["ICA [spectral_ica_components | braucht Standard/Center]"]
        n_spectra_nmf["NMF [spectral_nmf_components | nur roh: ohne SG/SNV/Skalierer/Indizes]"]
        n_spectra_osp["OSP [spectral_osp_components | nur roh: ohne SNV/Skalierer/Indizes]"]
        n_spectra_lpp["LPP [spectral_lpp_components + spectral_lpp_neighbors]"]
        n_spectra_vsrp["VSRP [spectral_vsrp_components]"]
        n_spectra_dbn["DBN [spectral_dbn_layers | braucht Min-Max]"]
        n_spectra_no_reduction["keine Reduktion [Standard]"]
    end
        d_concat["Merkmale (verkettet)"]
    X -.-> n_aez_aez
    X -.-> d_month
    d_month -->|"Standard"| n_month_cyclic
    d_month -->|"False"| n_month_month
    X --> n_albedo_drop_empty
    X --> n_indices_drop_empty
    X --> n_spectra_drop_empty
    X -.-> n_aez_month_aez_month
    n_albedo_drop_empty --> n_albedo_interpolate
    n_albedo_interpolate -.-> n_albedo_albedo
    n_albedo_albedo --> n_albedo_standard_scale
    n_albedo_standard_scale --> d_concat
    n_albedo_albedo --> n_albedo_minmax_scale
    n_albedo_minmax_scale --> d_concat
    n_albedo_albedo --> n_albedo_centering
    n_albedo_centering --> d_concat
    n_albedo_albedo --> n_albedo_no_scaler
    n_albedo_no_scaler --> d_concat
    n_indices_drop_empty --> n_indices_interpolate
    n_indices_interpolate -.-> n_indices_indices
    n_indices_indices --> n_indices_select
    n_indices_select --> n_indices_impute
    n_indices_impute -.-> n_indices_standard_scale
    n_indices_standard_scale --> d_concat
    n_indices_impute -.-> n_indices_minmax_scale
    n_indices_minmax_scale --> d_concat
    n_indices_impute -.-> n_indices_centering
    n_indices_centering --> d_concat
    n_indices_impute --> n_indices_no_scaler
    n_indices_no_scaler --> d_concat
    X -.-> n_principal_components_drop_empty
    n_principal_components_drop_empty -.-> n_principal_components_interpolate
    n_principal_components_interpolate -.-> n_principal_components_standard_scale
    n_principal_components_standard_scale -.-> n_principal_components_pca
    n_principal_components_pca --> d_concat
    n_spectra_drop_empty --> n_spectra_interpolate
    n_spectra_interpolate -.-> n_spectra_savgol
    n_spectra_savgol -.-> n_spectra_snv
    n_spectra_snv --> d_scaler
    d_scaler --> n_spectra_standard_scale
    n_spectra_standard_scale --> d_reduction
    d_scaler --> n_spectra_minmax_scale
    n_spectra_minmax_scale --> d_reduction
    d_scaler --> n_spectra_centering
    n_spectra_centering --> d_reduction
    d_scaler --> n_spectra_no_scaler
    n_spectra_no_scaler --> d_reduction
    d_reduction --> n_spectra_pca
    n_spectra_pca --> d_concat
    d_reduction --> n_spectra_ica
    n_spectra_ica --> d_concat
    d_reduction --> n_spectra_nmf
    n_spectra_nmf --> d_concat
    d_reduction --> n_spectra_osp
    n_spectra_osp --> d_concat
    d_reduction --> n_spectra_lpp
    n_spectra_lpp --> d_concat
    d_reduction --> n_spectra_vsrp
    n_spectra_vsrp --> d_concat
    d_reduction --> n_spectra_dbn
    n_spectra_dbn --> d_concat
    d_reduction --> n_spectra_no_reduction
    n_spectra_no_reduction --> d_concat
    n_aez_aez --> d_concat
    n_month_cyclic --> d_concat
    n_month_month --> d_concat
    n_albedo_albedo --> d_concat
    n_aez_month_aez_month --> d_concat
    n_indices_select --> d_concat
```
<!-- pipeline-options-diagram:end -->

### Dimensionsreduktion

Alle Verfahren fassen die 131 Bänder als letzter Spektral-Schritt zusammen; gleichzeitig ist
höchstens eines aktiv. Der Vergleich folgt acht unüberwachten Verfahren auf
Hyperspektraldaten ([Lupu et al.](https://www.researchgate.net/publication/390286182_Quick_unsupervised_hyperspectral_dimensionality_reduction_for_earth_observation_a_comparison)).

| Methode | Feld | Skalierung | Hinweis |
| --- | --- | --- | --- |
| PCA | `spectral_pca_components` | Standard-Scaler oder Zentrierung (`use_spectral_centering`) | Varianz-maximierende Komponenten |
| ICA | `spectral_ica_components` | Standard-Scaler oder Zentrierung | Statistisch unabhängige Komponenten (FastICA, whitened intern) |
| NMF | `spectral_nmf_components` | Keine (rohe Reflektanzen) | Multiplikative Updates; sklearn kennt nur NNDSVD/Random-Init |
| OSP | `spectral_osp_components` | Keine (rohe Spektren) | Automatische Zielgenerierung: iterative Endmember, Projektion darauf |
| LPP | `spectral_lpp_components` + `spectral_lpp_neighbors` | Beliebig | Nachbarschafts-erhaltend, lineare Projektion |
| VSRP | `spectral_vsrp_components` | Keine nötig | Mittelwert-zentrierte Achlioptas-Projektion (c = √d) |
| DBN | `spectral_dbn_layers=(…, k)` | Min-Max-Scaler nötig | Gestapelte Bernoulli-RBMs (nur Vortraining, ohne Fine-Tuning), Eingaben in [0, 1] |

Der CAE ist nicht umgesetzt – er braucht PyTorch/Keras (siehe Abschnitt 4 zum Wrapper-Muster).

### Befunde

Kein Verfahren gewinnt überall: PCA/ICA/OSP sind mit wenigen Bändern am besten (danach Plateau oder Abfall), LPP verbessert sich mit mehr Bändern weiter und klassifiziert am besten, OSP ist am robustesten gegen Streifen-Artefakte, VSRP ist am schnellsten. Für PCA/OSP genügen ca. 200 zufällige Pixel zum Fitten – bei unseren 3912 Trainingszeilen fitten wir trotzdem auf allen.

### Nicht umgesetzt (Ausblick)

- **CAE**: 1D-Convolution entlang der Wellenlänge mit linearer Engstelle – braucht PyTorch/Keras, siehe Abschnitt 4.
- **NMF mit OSP-Init**: sklearn kennt nur NNDSVD/Random-Init; eigene Init-Matrizen bräuchten eine angepasste NMF-Schleife.
- **DBN-Fine-Tuning**: fehlt (Momentum und L1-Strafe per Gradientenabstieg); die RBMs enden nach dem schichtweisen Vortraining.
- **Spalten-sampling-PCA**: reine Beschleunigung für 500k-Pixel-Szenen, bei unserer Datengröße überflüssig.

## 3. Tunen

Hyperparameter nur per Cross-Validation auf dem Train-Teil tunen – der konfigurierte Holdout ist
für den Endvergleich reserviert.

```python
from sklearn.model_selection import GridSearchCV

from awp2.data import load_folds
from awp2.evaluation import scorer

search = GridSearchCV(
    model,
    {"model__strategy": ["most_frequent", "stratified"]},
    scoring=scorer("bacc_combined"),
    cv=load_folds(),
)
search.fit(split.X_train, split.y_train)
```

Modelle einheitlich ausführen und in MLflow vergleichen: [Experimente & MLflow](../entwicklung/tracking.md).

`scoring=scorer(...)` ist nötig, weil sklearns Standard-Score keine zwei Zielspalten kennt.
Für Modelle ohne `class_weight` gegen das Klassenungleichgewicht:
[`balanced_sample_weight`][awp2.data.split.balanced_sample_weight].

## 4. Modelle außerhalb von sklearn (z. B. neuronale Netze)

Daten, Split, Folds und `evaluate()` hängen an keinem Framework. Nur `run()`, `tune()` und
`build_preprocessor()` erwarten die **sklearn-Schnittstelle** – die hat aber jedes Modell, das
in eine kleine Hülle (Wrapper) gepackt wird. Ein Netz in PyTorch, Keras o. Ä. läuft dann
unverändert durch die Pipeline und wird in MLflow protokolliert:

```python
from sklearn.base import BaseEstimator, ClassifierMixin


class SpectralLSTM(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_size: int = 64, epochs: int = 50) -> None:
        self.hidden_size = hidden_size  # Hyperparameter nur speichern → tune() kann sie setzen
        self.epochs = epochs

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "SpectralLSTM":
        ...  # Labels kodieren, Netz bauen und trainieren
        return self

    def predict(
        self, X: pd.DataFrame
    ) -> np.ndarray: ...  # Array der Form (n, 2): Spalten Crop, Stage
```

| Worauf achten | Warum |
| --- | --- |
| Alle Hyperparameter als Argumente von `__init__`, dort nur speichern | `tune()` kopiert das Modell mit `clone()` und setzt Werte über diese Namen |
| `X` ist ein DataFrame aus Bändern **und** ggf. `AEZ`/`Month` | Für ein LSTM/1D-CNN die Bandspalten zur Sequenz `(n, Bänder, 1)` umformen; Metadaten getrennt einspeisen oder `use_meta=False` |
| Skalierung einschalten (`PreprocessingConfig(scale=True)`) | Netze lernen auf unskalierten Reflektanzen schlecht |
| Seed aus `awp2.config` im Framework setzen | Sonst ist jeder Lauf anders und Vergleiche sind wertlos |
| Kleine Grids, `log_model=False` beim Ausprobieren | `tune()` trainiert jeden Kandidaten 5-mal; das Speichern großer Netze kostet Zeit |

Für PyTorch nimmt [skorch](https://skorch.readthedocs.io/) den Trainings-Code ab; die zwei
Zielspalten muss der Wrapper trotzdem selbst behandeln. Wer die Hülle nicht bauen will, kann
auch ohne `run()` arbeiten: `load_split()`, Vorverarbeitung auf `X_train` fitten, eigenes
Training, dann `evaluate()` – muss Tracking und das Verbot, auf der Validierung zu tunen, dann
aber selbst sicherstellen.
