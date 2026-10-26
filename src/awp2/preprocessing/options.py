"""Options for the shared preprocessing pipeline."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from awp2.config import (
    F_CLASSIF_SCORE,
    HYPERION_SENSOR,
    SENTINEL2_SENSOR,
    SPECTRAL_REGIONS,
    BandScore,
    Sensor,
)


class PreprocessingConfig(BaseModel):
    """Options of the standard preprocessing.

    Preprocessing steps are configured here and implemented in the corresponding
    transformer module, rather than as ad-hoc notebook code.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    spectral_regions: tuple[str, ...] | None = Field(
        default=None,
        description="Restrict every spectral branch to these SPECTRAL_REGIONS before "
        "interpolation. None keeps all bands; () gives a metadata-only baseline.",
    )
    use_meta: bool = Field(
        default=True, description="Use AEZ (one-hot) and Month as features next to the spectrum."
    )
    use_spectra: bool = Field(
        default=True,
        description="Use the spectral bands as features; False keeps only the other enabled "
        "groups (meta, indices, albedo, PC_1/PC_2), e.g. for an indices-only model.",
    )
    spectral_sensor: Sensor = Field(
        default=HYPERION_SENSOR,
        description="Spectral sampling of the input: hyperion keeps all bands, sentinel2 "
        "averages them into the 10 Sentinel-2 surface bands (after interpolation).",
    )
    spectral_band_step: int | None = Field(
        default=None,
        ge=2,
        description="Keep only every n-th usable band (after interpolation, smoothing and SNV) "
        "to simulate a coarser spectral resolution; None keeps all bands.",
    )
    use_categorical_context: bool = Field(
        default=False,
        description=(
            "Keep AEZ and Month as unscaled categorical columns for a model that learns "
            "context embeddings; overrides one-hot AEZ and cyclic Month encoding."
        ),
    )
    scale: bool = Field(
        default=False,
        description="Deprecated, will be removed soon – use use_spectral_standard_scale instead. "
        "Kept for old runs: standardises spectra and month.",
    )
    use_spectral_standard_scale: bool = Field(
        default=False,
        description="Standardise spectral bands only (per-band mean/std from the training part); "
        "leaves AEZ and Month untouched.",
    )
    use_spectral_minmax_scale: bool = Field(
        default=False,
        description="Scale spectral bands to [0, 1] per band from the training part; "
        "leaves AEZ and Month untouched.",
    )
    use_spectral_snv: bool = Field(
        default=False,
        description="Apply Standard Normal Variate scatter correction per spectrum (zero mean, "
        "unit variance across bands); applied after interpolation, before indices and scaling.",
    )
    use_spectral_centering: bool = Field(
        default=False,
        description="Subtract the per-band training mean without variance scaling "
        "(paper-faithful PCA input); mutually exclusive with the scalers.",
    )
    use_mean_albedo: bool = Field(
        default=False,
        description="Append the per-spectrum mean reflectance (mean albedo, computed "
        "after interpolation) as an extra feature; preserves the brightness that SNV removes. "
        "Scaled with the same method as the spectra.",
    )
    use_savgol: bool = Field(
        default=False,
        description="Smooth spectra with a Savitzky-Golay filter after interpolation, one "
        "wavelength segment at a time.",
    )
    savgol_window_length: int | None = Field(
        default=None,
        description="Savitzky-Golay window size in bands (odd); required with use_savgol.",
    )
    savgol_polyorder: int | None = Field(
        default=None,
        description="Savitzky-Golay polynomial degree; required with use_savgol.",
    )
    spectral_pca_components: int | None = Field(
        default=None,
        gt=0,
        description="Reduce the spectral bands to this many principal components; "
        "None keeps all bands. Requires use_spectral_standard_scale or use_spectral_centering.",
    )
    spectral_select_k: int | None = Field(
        default=None,
        gt=0,
        description="Keep the k bands with the highest univariate score against the crop/stage "
        "label, fitted per training fold; None keeps all bands.",
    )
    spectral_select_score: BandScore = Field(
        default=F_CLASSIF_SCORE,
        description="Band score for spectral_select_k: f_classif (ANOVA F) or mutual_info.",
    )
    spectral_ica_components: int | None = Field(
        default=None,
        gt=0,
        description="Independent components of the spectral bands; "
        "None disables the method. Requires use_spectral_standard_scale or use_spectral_centering.",
    )
    spectral_nmf_components: int | None = Field(
        default=None,
        gt=0,
        description="Non-negative parts of the raw spectra (multiplicative updates); "
        "None disables the method. Needs non-negative input: must not combine with "
        "savgol, snv, scalers or indices.",
    )
    spectral_osp_components: int | None = Field(
        default=None,
        gt=0,
        description="Automatic target generation: iterative endmember signatures of the raw "
        "spectra; None disables the method. Must not combine with snv, scalers or indices.",
    )
    spectral_lpp_components: int | None = Field(
        default=None,
        gt=0,
        description="Locality-preserving components of the spectral bands; "
        "None disables the method. Requires spectral_lpp_neighbors.",
    )
    spectral_lpp_neighbors: int | None = Field(
        default=None,
        gt=0,
        description="Neighbours per spectrum for the LPP adjacency graph; "
        "required with spectral_lpp_components.",
    )
    spectral_vsrp_components: int | None = Field(
        default=None,
        gt=0,
        description="Very sparse random projection of mean-centered spectra; "
        "None disables the method. Needs no scaler.",
    )
    spectral_dbn_layers: tuple[int, ...] | None = Field(
        default=None,
        description="Stacked BernoulliRBM architecture, e.g. (64, 20); the last entry is the "
        "output size. Requires use_spectral_minmax_scale for [0, 1] input.",
    )
    use_vegetation_indices: bool = Field(
        default=False,
        description="Compute the reproducible kickoff vegetation-index catalogue from interpolated "
        "bands in a parallel branch; ratios need physical reflectance, so SNV must not feed them.",
    )
    use_principal_components: bool = Field(
        default=False,
        description="Append PC_1 and PC_2 from a PCA fitted on standardized interpolated bands "
        "in a parallel branch.",
    )
    use_aez_month: bool = Field(
        default=False,
        description="One-hot encode each AEZ × month combination seen in training next to the "
        "separate AEZ and month features; requires use_meta.",
    )
    use_acquisition_centering: bool = Field(
        default=False,
        description="Subtract the mean training spectrum of each AEZ x month acquisition from "
        "the spectra before all other spectral steps; unseen acquisitions use the overall mean.",
    )
    use_cyclic_month: bool = Field(
        default=True,
        description="Replace numeric Month with sine/cosine annual-cycle features.",
    )
    max_interpolation_gap_nm: int = Field(
        default=15,
        gt=0,
        description=(
            "Maximum wavelength span for linear interpolation; wider gaps use the nearest "
            "measured band."
        ),
    )

    @model_validator(mode="after")
    def _check_options(self) -> Self:
        """Reject ambiguous or incomplete preprocessing combinations.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If ``use_aez_month`` is set without ``use_meta`` or with
                ``use_categorical_context``; if ``spectral_regions`` repeats or does not know a
                region, is empty without ``use_meta`` or with a spectral feature branch; if more
                than one reduction method is selected; if centering,
                scalers, PCA, ICA or the deprecated ``scale`` flag are combined ambiguously;
                if NMF, OSP or DBN get unsuitable input; if Savitzky-Golay or LPP
                arguments are set without their method or missing with it; or if
                ``use_spectra=False`` keeps spectral steps or leaves no features.
        """
        if not self.use_spectra:
            self._check_without_spectra()
        if self.use_acquisition_centering and (
            self.spectral_nmf_components is not None or self.spectral_osp_components is not None
        ):
            raise ValueError("acquisition centering must not combine with NMF or OSP (raw spectra)")
        if self.spectral_sensor == SENTINEL2_SENSOR and (
            self.use_savgol or self.spectral_band_step is not None
        ):
            raise ValueError(
                "spectral_sensor='sentinel2' must not combine with savgol or spectral_band_step"
            )
        if self.spectral_regions is not None:
            unknown = set(self.spectral_regions) - {region.name for region in SPECTRAL_REGIONS}
            if unknown:
                raise ValueError(f"spectral_regions requires known names; unknown: {unknown}")
            if len(set(self.spectral_regions)) != len(self.spectral_regions):
                raise ValueError("spectral_regions requires unique names")
            if not self.spectral_regions:
                if not self.use_meta:
                    raise ValueError("Empty spectral_regions requires use_meta=True")
                if (
                    self.use_mean_albedo
                    or self.use_vegetation_indices
                    or self.use_principal_components
                ):
                    raise ValueError("Spectral feature branches require nonempty spectral_regions")
        if self.use_aez_month and (not self.use_meta or self.use_categorical_context):
            raise ValueError("use_aez_month requires use_meta without use_categorical_context")
        if self.use_spectral_standard_scale and self.use_spectral_minmax_scale:
            raise ValueError("use only one of use_spectral_standard_scale/minmax_scale")
        if self.use_spectral_centering and (
            self.scale or self.use_spectral_standard_scale or self.use_spectral_minmax_scale
        ):
            raise ValueError("use_spectral_centering must not combine with scalers")
        if self.scale and (self.use_spectral_standard_scale or self.use_spectral_minmax_scale):
            raise ValueError("deprecated 'scale' must not be combined with the spectral scalers")
        reductions = [
            self.spectral_pca_components,
            self.spectral_select_k,
            self.spectral_ica_components,
            self.spectral_nmf_components,
            self.spectral_osp_components,
            self.spectral_lpp_components,
            self.spectral_vsrp_components,
        ]
        if (
            sum(selected is not None for selected in reductions)
            + (self.spectral_dbn_layers is not None)
            > 1
        ):
            raise ValueError("use at most one dimensionality reduction method")
        needs_centering = (
            self.spectral_pca_components,
            self.spectral_ica_components,
        )
        if any(selected is not None for selected in needs_centering):
            if not (self.use_spectral_standard_scale or self.use_spectral_centering):
                raise ValueError(
                    "pca/ica reduction requires use_spectral_standard_scale or "
                    "use_spectral_centering"
                )
        if self.spectral_nmf_components is not None and (
            self.scale
            or self.use_spectral_standard_scale
            or self.use_spectral_minmax_scale
            or self.use_spectral_centering
            or self.use_spectral_snv
            or self.use_savgol
            or self.use_vegetation_indices
        ):
            raise ValueError(
                "nmf reduction needs raw spectra: must not combine with savgol, snv, "
                "scalers or vegetation indices"
            )
        if self.spectral_osp_components is not None and (
            self.scale
            or self.use_spectral_standard_scale
            or self.use_spectral_minmax_scale
            or self.use_spectral_centering
            or self.use_spectral_snv
            or self.use_vegetation_indices
        ):
            raise ValueError(
                "osp reduction needs raw spectra: must not combine with snv, "
                "scalers or vegetation indices"
            )
        if self.spectral_lpp_components is None and self.spectral_lpp_neighbors is not None:
            raise ValueError("spectral_lpp_neighbors requires spectral_lpp_components")
        if self.spectral_lpp_components is not None and self.spectral_lpp_neighbors is None:
            raise ValueError("spectral_lpp_components requires spectral_lpp_neighbors")
        if self.spectral_lpp_components is not None and self.use_vegetation_indices:
            raise ValueError("lpp reduction must not combine with vegetation indices")
        if self.spectral_dbn_layers is not None:
            if not self.use_spectral_minmax_scale:
                raise ValueError("dbn reduction requires use_spectral_minmax_scale=True")
            if self.use_vegetation_indices:
                raise ValueError("dbn reduction must not combine with vegetation indices")
            if len(self.spectral_dbn_layers) == 0 or any(
                size < 1 for size in self.spectral_dbn_layers
            ):
                raise ValueError("spectral_dbn_layers must hold at least one positive size")
        if self.use_savgol:
            if self.savgol_window_length is None or self.savgol_polyorder is None:
                raise ValueError("use_savgol requires savgol_window_length and savgol_polyorder")
            if self.savgol_window_length < 3 or self.savgol_window_length % 2 == 0:
                raise ValueError("savgol_window_length must be odd and at least 3")
            if not 0 <= self.savgol_polyorder < self.savgol_window_length:
                raise ValueError("savgol_polyorder must satisfy 0 <= polyorder < window_length")
        elif self.savgol_window_length is not None or self.savgol_polyorder is not None:
            raise ValueError("savgol_window_length/polyorder require use_savgol=True")
        return self

    def _check_without_spectra(self) -> None:
        spectral_steps = {
            "use_savgol": self.use_savgol,
            "use_spectral_snv": self.use_spectral_snv,
            "spectral_sensor": self.spectral_sensor != HYPERION_SENSOR,
            "spectral_band_step": self.spectral_band_step is not None,
            "spectral_pca_components": self.spectral_pca_components is not None,
            "spectral_select_k": self.spectral_select_k is not None,
            "spectral_ica_components": self.spectral_ica_components is not None,
            "spectral_nmf_components": self.spectral_nmf_components is not None,
            "spectral_osp_components": self.spectral_osp_components is not None,
            "spectral_lpp_components": self.spectral_lpp_components is not None,
            "spectral_vsrp_components": self.spectral_vsrp_components is not None,
            "spectral_dbn_layers": self.spectral_dbn_layers is not None,
        }
        enabled = [name for name, active in spectral_steps.items() if active]
        if enabled:
            raise ValueError(f"use_spectra=False must not combine with spectral steps: {enabled}")
        if not (
            self.use_meta
            or self.use_vegetation_indices
            or self.use_mean_albedo
            or self.use_principal_components
        ):
            raise ValueError("use_spectra=False needs another feature group")
