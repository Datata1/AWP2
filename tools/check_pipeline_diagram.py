"""CI check for the pipeline options diagram: freshness and step coverage."""

from pathlib import Path

from sklearn.pipeline import Pipeline

from awp2.preprocessing import (
    PreprocessingConfig,
    _diagram_steps,
    build_preprocessor,
    pipeline_options_diagram,
)

PAGE = Path("docs/daten/pipeline.md")
START = "<!-- pipeline-options-diagram:start -->"
END = "<!-- pipeline-options-diagram:end -->"


def _probe_configs() -> list[PreprocessingConfig]:
    """Valid configs covering every optional preprocessing step (no fitting needed)."""
    return [
        PreprocessingConfig(),
        PreprocessingConfig(
            use_mean_albedo=True,
            use_savgol=True,
            savgol_window_length=11,
            savgol_polyorder=3,
            use_spectral_snv=True,
            use_vegetation_indices=True,
            use_spectral_standard_scale=True,
            spectral_pca_components=5,
        ),
        PreprocessingConfig(
            use_mean_albedo=True, use_spectral_minmax_scale=True, spectral_dbn_layers=(64, 20)
        ),
        PreprocessingConfig(
            use_mean_albedo=True, use_spectral_centering=True, spectral_ica_components=5
        ),
        PreprocessingConfig(use_meta=False, spectral_nmf_components=5),
        PreprocessingConfig(spectral_osp_components=5),
        PreprocessingConfig(spectral_lpp_components=5, spectral_lpp_neighbors=5),
        PreprocessingConfig(spectral_vsrp_components=5),
        PreprocessingConfig(use_cyclic_month=False),
        PreprocessingConfig(use_principal_components=True),
        PreprocessingConfig(use_aez_month=True),
        PreprocessingConfig(spectral_band_step=2),
        PreprocessingConfig(use_spectra=False, use_vegetation_indices=True),
    ]


def main() -> int:
    """Check diagram freshness and coverage; return the exit code."""
    page = PAGE.read_text(encoding="utf-8")
    section = page.split(START)[1].split(END)[0].strip()
    if section != pipeline_options_diagram().strip():
        print(
            "diagram stale: regenerate (see 'Ablauf der Vorverarbeitung' in docs/daten/pipeline.md)"
        )
        return 1
    covered = _diagram_steps()
    missing: set[tuple[str, str]] = set()
    for config in _probe_configs():
        for branch, transformer, _columns in build_preprocessor(config).transformers:
            if isinstance(transformer, Pipeline):
                missing.update((branch, name) for name, _step in transformer.steps)
            else:
                missing.add((branch, branch))
    missing -= covered
    if missing:
        print(f"steps missing from options diagram: {sorted(missing)}")
        return 1
    print(f"diagram fresh; {len(covered)} diagram steps cover all probed builds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
