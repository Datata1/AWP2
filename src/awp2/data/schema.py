"""pandera schemas that validate the raw data on load."""

import pandera.pandas as pa

from awp2.config import BAND_PREFIX, CROPS, ID_COL, STAGES

feature_schema = pa.DataFrameSchema(
    {
        ID_COL: pa.Column(str, unique=True),
        "AEZ": pa.Column(int, pa.Check.in_range(1, 20)),
        "Month": pa.Column(int, pa.Check.in_range(1, 12)),
        rf"^{BAND_PREFIX}\d+$": pa.Column(float, pa.Check.ge(0), nullable=True, regex=True),
    },
    strict=True,
)

train_schema = feature_schema.add_columns(
    {
        "Crop": pa.Column(str, pa.Check.isin(CROPS)),
        "Stage": pa.Column(str, pa.Check.isin(STAGES)),
    }
)
