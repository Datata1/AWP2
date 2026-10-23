"""Shared sklearn preprocessing with stable public imports."""

from awp2.preprocessing.bands import DropEmptyBands as DropEmptyBands
from awp2.preprocessing.bands import InterpolateBands as InterpolateBands
from awp2.preprocessing.bands import MeanAlbedo as MeanAlbedo
from awp2.preprocessing.bands import SavitzkyGolaySmoothing as SavitzkyGolaySmoothing
from awp2.preprocessing.bands import StandardNormalVariate as StandardNormalVariate
from awp2.preprocessing.bands import _BandTransformer as _BandTransformer
from awp2.preprocessing.decomposition import DeepBeliefFeatures as DeepBeliefFeatures
from awp2.preprocessing.decomposition import (
    LocalityPreservingProjection as LocalityPreservingProjection,
)
from awp2.preprocessing.decomposition import (
    OrthogonalSubspaceProjection as OrthogonalSubspaceProjection,
)
from awp2.preprocessing.decomposition import (
    PrincipalComponentFeatures as PrincipalComponentFeatures,
)
from awp2.preprocessing.decomposition import (
    VerySparseRandomProjection as VerySparseRandomProjection,
)
from awp2.preprocessing.indices import AddVegetationIndices as AddVegetationIndices
from awp2.preprocessing.indices import SelectVegetationIndices as SelectVegetationIndices
from awp2.preprocessing.indices import ndvi as ndvi
from awp2.preprocessing.indices import nearest_band as nearest_band
from awp2.preprocessing.metadata import AezMonthEncoder as AezMonthEncoder
from awp2.preprocessing.metadata import CyclicMonthEncoder as CyclicMonthEncoder
from awp2.preprocessing.options import PreprocessingConfig as PreprocessingConfig
from awp2.preprocessing.pipeline import _region_columns as _region_columns
from awp2.preprocessing.pipeline import build_preprocessor as build_preprocessor
from awp2.preprocessing.reporting import describe_preprocessor as describe_preprocessor

__all__ = [
    "ndvi",
    "nearest_band",
    "DropEmptyBands",
    "InterpolateBands",
    "SavitzkyGolaySmoothing",
    "StandardNormalVariate",
    "OrthogonalSubspaceProjection",
    "VerySparseRandomProjection",
    "LocalityPreservingProjection",
    "DeepBeliefFeatures",
    "AddVegetationIndices",
    "SelectVegetationIndices",
    "PrincipalComponentFeatures",
    "CyclicMonthEncoder",
    "MeanAlbedo",
    "AezMonthEncoder",
    "PreprocessingConfig",
    "build_preprocessor",
    "describe_preprocessor",
]
