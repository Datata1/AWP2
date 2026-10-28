"""AutoML winner: combined RBF-SVM on smoothed, MinMax-scaled spectra, trained on all labelled data.

Hyperparameters and preprocessing are frozen from MLflow run c99e3915 (`automl_final`, CV search
a08557a2, validation BAcc combined 0.883).

Usage:
    uv run python tools/predict_automl_final.py [--test-file PATH] [--output PATH]
"""

from awp2.models import make_combined_svm
from awp2.preprocessing import PreprocessingConfig
from awp2.submission import main

NAME = "automl_final"
SVM_C = 100.0
SVM_GAMMA = "scale"
INTERPOLATION_GAP_NM = 10
SAVGOL_WINDOW = 11
SAVGOL_POLYORDER = 3

if __name__ == "__main__":
    main(
        NAME,
        make_combined_svm().set_params(estimator__C=SVM_C, estimator__gamma=SVM_GAMMA),
        PreprocessingConfig(
            use_spectral_minmax_scale=True,
            use_savgol=True,
            savgol_window_length=SAVGOL_WINDOW,
            savgol_polyorder=SAVGOL_POLYORDER,
            use_mean_albedo=True,
            use_vegetation_indices=True,
            use_aez_month=True,
            max_interpolation_gap_nm=INTERPOLATION_GAP_NM,
        ),
    )
