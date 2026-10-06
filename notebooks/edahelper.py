import pandas as pd


def fill_spectral_nans(
    df: pd.DataFrame,
    base_step_range: int = 15,
    total_range: int = 30,
) -> pd.DataFrame:
    """Fill missing X<wavelength> values using neighboring spectral columns."""
    if base_step_range <= 0:
        raise ValueError("base_step_range must be positive")
    if total_range < base_step_range:
        raise ValueError("total_range must be at least base_step_range")

    bands = sorted(
        (
            (int(column[1:]), column)
            for column in df.columns
            if column.startswith("X") and column[1:].isdigit()
        ),
        key=lambda item: item[0],
    )
    wavelengths = [wavelength for wavelength, _ in bands]
    band_names = [column for _, column in bands]

    original = df[band_names].copy()
    result = df.copy()
    column_means = original.mean()

    def nearest_value(row_number: int, candidate_indices: list[int]) -> float | None:
        for candidate_index in candidate_indices:
            value = original.iat[row_number, candidate_index]
            if pd.notna(value):
                return value
        return None

    for column_index, column_name in enumerate(band_names):
        for row_number in range(len(original)):
            if pd.notna(original.iat[row_number, column_index]):
                continue

            radius = base_step_range
            filled = False

            while True:
                left_indices = [
                    index
                    for index in range(column_index - 1, -1, -1)
                    if wavelengths[column_index] - wavelengths[index] <= radius
                ]
                right_indices = [
                    index
                    for index in range(column_index + 1, len(band_names))
                    if wavelengths[index] - wavelengths[column_index] <= radius
                ]

                left_value = nearest_value(row_number, left_indices)
                right_value = nearest_value(row_number, right_indices)

                if left_value is not None or right_value is not None:
                    if left_value is not None and right_value is not None:
                        value = (left_value + right_value) / 2
                    else:
                        value = left_value if left_value is not None else right_value

                    result.iat[row_number, result.columns.get_loc(column_name)] = value
                    filled = True
                    break

                if radius >= total_range:
                    break
                radius = min(radius + base_step_range, total_range)

            if not filled:
                result.iat[row_number, result.columns.get_loc(column_name)] = column_means[
                    column_name
                ]

    return result
