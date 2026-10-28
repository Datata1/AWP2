"""Download and read third-party reference data in ``data/external/`` (not in git)."""

import hashlib
import re
import urllib.request
from pathlib import Path

import pandas as pd

from awp2.config import (
    DOWNLOAD_TIMEOUT_S,
    SENTINEL2_SRF_BAND_PATTERN,
    SENTINEL2_SRF_FILE,
    SENTINEL2_SRF_SHA256,
    SENTINEL2_SRF_SHEET,
    SENTINEL2_SRF_URL,
    SENTINEL2_SRF_WAVELENGTH_COL,
    Sentinel2Platform,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_sentinel2_srf(
    path: Path = SENTINEL2_SRF_FILE,
    url: str = SENTINEL2_SRF_URL,
    sha256: str = SENTINEL2_SRF_SHA256,
) -> Path:
    """Download the ESA Sentinel-2 spectral response functions, verified by checksum.

    A file that is already present with the expected checksum is kept, so repeated calls
    need no network access.

    Args:
        path: Target file.
        url: Source of the SRF workbook (no login needed).
        sha256: Expected SHA-256 of the workbook; pins the SRF version.

    Returns:
        Path of the verified workbook.

    Raises:
        ValueError: If the downloaded file does not match ``sha256``; nothing is written then.
    """
    if path.exists() and _sha256(path) == sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
        partial.write_bytes(response.read())
    actual = _sha256(partial)
    if actual != sha256:
        partial.unlink()
        raise ValueError(f"checksum mismatch for {url}: expected {sha256}, got {actual}")
    partial.replace(path)
    return path


def load_sentinel2_srf(
    platform: Sentinel2Platform, path: Path = SENTINEL2_SRF_FILE
) -> pd.DataFrame:
    """Read the spectral response functions of one Sentinel-2 satellite.

    Args:
        platform: Satellite whose responses to read.
        path: Workbook from `download_sentinel2_srf`.

    Returns:
        Relative response (0–1) per band name (``B1`` … ``B12``, ``B8A``), indexed by wavelength
            in nm.
    """
    srf = pd.read_excel(path, sheet_name=SENTINEL2_SRF_SHEET.format(platform=platform))
    srf = srf.set_index(SENTINEL2_SRF_WAVELENGTH_COL)
    return srf.rename(columns=lambda c: re.sub(SENTINEL2_SRF_BAND_PATTERN, r"\1", str(c)))


def main() -> None:
    """Download all external reference data (``make external``)."""
    print(download_sentinel2_srf())


if __name__ == "__main__":
    main()
