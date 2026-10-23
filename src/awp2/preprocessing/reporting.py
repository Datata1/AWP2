"""Reporting for the shared preprocessing pipeline."""

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import NMF, PCA, FastICA
from sklearn.pipeline import Pipeline

from awp2.preprocessing.bands import InterpolateBands, SavitzkyGolaySmoothing
from awp2.preprocessing.decomposition import (
    DeepBeliefFeatures,
    LocalityPreservingProjection,
    OrthogonalSubspaceProjection,
    PrincipalComponentFeatures,
    VerySparseRandomProjection,
)
from awp2.preprocessing.indices import SelectVegetationIndices


def _branch_steps(
    name: str, transformer: BaseEstimator | str
) -> list[tuple[str, BaseEstimator | str]]:
    if isinstance(transformer, Pipeline):
        return list(transformer.steps)
    return [(name, transformer)]


def _describe_step(name: str, step: BaseEstimator | str) -> str:
    if name == "context":
        return "AEZ and Month as categorical context"
    if isinstance(step, str):
        return f"Month {step}"
    known: dict[str, str] = {
        "drop_empty": "Drop empty bands",
        "snv": "SNV scatter correction per spectrum",
        "indices": "Vegetation-index catalogue (from interpolated bands)",
        "standard_scale": "Standardize bands (per-band mean and std)",
        "minmax_scale": "Min-max scale bands to [0, 1]",
        "centering": "Center bands (per-band mean)",
        "aez": "One-hot encode AEZ",
        "albedo": "Mean albedo per spectrum (brightness after interpolation, before SNV)",
        "cyclic": "Cyclic month encoding (sin/cos)",
        "month": "Standardize month",
        "scale": "Standardize month",
        "aez_month": "One-hot encode AEZ x month combinations",
    }
    if name in known:
        return known[name]
    if name == "interpolate" and isinstance(step, InterpolateBands):
        return f"Interpolate band gaps (max {step.max_interpolation_gap_nm} nm)"
    if name == "select" and isinstance(step, SelectVegetationIndices):
        return "Keep only the vegetation indices"
    if name == "savgol" and isinstance(step, SavitzkyGolaySmoothing):
        return (
            f"Savitzky-Golay smoothing (window {step.window_length} bands, "
            f"polyorder {step.polyorder})"
        )
    if name == "pca" and isinstance(step, PCA):
        return f"PCA to {step.n_components} components"
    if name == "pca" and isinstance(step, PrincipalComponentFeatures):
        return "PCA to PC_1 and PC_2"
    if name == "ica" and isinstance(step, FastICA):
        return f"ICA to {step.n_components} components"
    if name == "nmf" and isinstance(step, NMF):
        return f"NMF to {step.n_components} components"
    if name == "osp" and isinstance(step, OrthogonalSubspaceProjection):
        return f"OSP to {step.n_components} endmembers"
    if name == "lpp" and isinstance(step, LocalityPreservingProjection):
        return f"LPP to {step.n_components} components ({step.n_neighbors} neighbors)"
    if name == "vsrp" and isinstance(step, VerySparseRandomProjection):
        return f"VSRP to {step.n_components} components"
    if name == "dbn" and isinstance(step, DeepBeliefFeatures):
        layers = "x".join(str(size) for size in step.hidden_layers)
        return f"DBN features ({layers})"
    return type(step).__name__


_BRANCH_LETTERS: dict[str, str] = {
    "aez": "A",
    "context": "A",
    "month": "B",
    "albedo": "C",
    "indices": "D",
    "spectra": "E",
    "aez_month": "F",
    "principal_components": "G",
}


def describe_preprocessor(preprocessor: ColumnTransformer) -> list[str]:
    r"""List the active preprocessing steps, grouped by independent branch.

    Reads the built transformer, so optional steps appear only when enabled. Each
    branch has a fixed letter (A: AEZ, B: Month, C: Albedo, D: Indices, E: Spectra,
    F: AEZ x month, G: PC_1 and PC_2)
    and steps are numbered per branch from 1, so the labels stay comparable across
    runs. Print the lines in the notebook to log which preprocessing a run used.

    Args:
        preprocessor: Built by ``build_preprocessor`` (fitting is not required).

    Returns:
        One labeled description per active step, e.g. ``"E3. PCA to 20 components"``.

    Raises:
        ValueError: If a branch has no letter yet (add one to ``_BRANCH_LETTERS``).

    Example:
        ```python
        print("\\n".join(describe_preprocessor(build_preprocessor(
            PreprocessingConfig(use_meta=True, use_spectral_snv=True)
        ))))
        ```
    """
    lines: list[str] = []
    for branch_name, transformer, _columns in preprocessor.transformers:
        if branch_name not in _BRANCH_LETTERS:
            raise ValueError(f"describe_preprocessor needs a letter for branch: {branch_name}")
        letter = _BRANCH_LETTERS[branch_name]
        steps = _branch_steps(branch_name, transformer)
        for number, (step_name, step) in enumerate(steps, start=1):
            lines.append(f"{letter}{number}. {_describe_step(step_name, step)}")
    return lines
