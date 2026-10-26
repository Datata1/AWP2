import hashlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from awp2.config import (
    SENTINEL2_SRF_SHEET,
    SENTINEL2_SRF_WAVELENGTH_COL,
    SENTINEL2A_PLATFORM,
    SENTINEL2B_PLATFORM,
)
from awp2.data import external

URL = "https://example.org/srf.xlsx"


def _fake_urlopen(payload: bytes) -> object:
    return patch.object(external.urllib.request, "urlopen", return_value=io.BytesIO(payload))


class DownloadSentinel2SrfTest(unittest.TestCase):
    def test_writes_file_when_checksum_matches(self) -> None:
        payload = b"srf workbook"
        with TemporaryDirectory() as tmp, _fake_urlopen(payload):
            target = Path(tmp) / "external" / "srf.xlsx"

            path = external.download_sentinel2_srf(target, URL, hashlib.sha256(payload).hexdigest())

            self.assertEqual(path.read_bytes(), payload)

    def test_rejects_checksum_mismatch_without_leaving_a_file(self) -> None:
        with TemporaryDirectory() as tmp, _fake_urlopen(b"tampered"):
            target = Path(tmp) / "srf.xlsx"

            with self.assertRaises(ValueError):
                external.download_sentinel2_srf(target, URL, hashlib.sha256(b"srf").hexdigest())

            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_keeps_verified_file_without_network(self) -> None:
        payload = b"srf workbook"
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "srf.xlsx"
            target.write_bytes(payload)
            with patch.object(external.urllib.request, "urlopen") as urlopen:
                external.download_sentinel2_srf(target, URL, hashlib.sha256(payload).hexdigest())

            urlopen.assert_not_called()


class LoadSentinel2SrfTest(unittest.TestCase):
    def test_reads_platform_sheet_with_plain_band_names(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "srf.xlsx"
            with pd.ExcelWriter(path) as writer:
                for platform, response in ((SENTINEL2A_PLATFORM, 1.0), (SENTINEL2B_PLATFORM, 0.5)):
                    pd.DataFrame(
                        {
                            SENTINEL2_SRF_WAVELENGTH_COL: [864, 865],
                            f"{platform}_SR_AV_B8A": [0.9, response],
                        }
                    ).to_excel(
                        writer,
                        sheet_name=SENTINEL2_SRF_SHEET.format(platform=platform),
                        index=False,
                    )

            srf = external.load_sentinel2_srf(SENTINEL2B_PLATFORM, path)

        self.assertEqual(list(srf.columns), ["B8A"])
        self.assertEqual(srf.loc[865, "B8A"], 0.5)


if __name__ == "__main__":
    unittest.main()
