"""pandera schemas that validate the raw data on load."""

import pandera.pandas as pa

from awp2.config import (
    AEZ_COL,
    AEZ_RANGE,
    BAND_PATTERN,
    CROP_COL,
    CROPS,
    ID_COL,
    MONTH_COL,
    MONTH_RANGE,
    STAGE_COL,
    STAGES,
)

feature_schema = pa.DataFrameSchema(
    {
        ID_COL: pa.Column(str, unique=True),
        AEZ_COL: pa.Column(int, pa.Check.in_range(*AEZ_RANGE)),
        MONTH_COL: pa.Column(int, pa.Check.in_range(*MONTH_RANGE)),
        BAND_PATTERN: pa.Column(float, pa.Check.ge(0), nullable=True, regex=True),
    },
    strict=True,
)

train_schema = feature_schema.add_columns(
    {
        CROP_COL: pa.Column(str, pa.Check.isin(CROPS)),
        STAGE_COL: pa.Column(str, pa.Check.isin(STAGES)),
    }
)
