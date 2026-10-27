"""Official best model: combined RBF-SVM with AEZ x month indicators, trained on all labelled data.

Hyperparameters and preprocessing are frozen from MLflow run 17dc9e16 (`combined_svm_aez_month`,
tuning `context_aez_month` d93cea89, validation BAcc combined 0.869).

Usage:
    uv run python tools/predict_combined_svm_aez_month.py [--test-file PATH] [--output PATH]
"""

from awp2.config import SEED
from awp2.models import make_combined_svm
from awp2.preprocessing import PreprocessingConfig
from awp2.submission import main

NAME = "combined_svm_aez_month"
SVM_C = 300.0
SVM_GAMMA = 0.003

if __name__ == "__main__":
    main(
        NAME,
        make_combined_svm().set_params(
            estimator__C=SVM_C, estimator__gamma=SVM_GAMMA, estimator__random_state=SEED
        ),
        PreprocessingConfig(use_spectral_standard_scale=True, use_aez_month=True),
    )
