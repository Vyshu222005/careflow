"""Download the Sepsis Cases event log (4TU.ResearchData) into data/raw.

Usage:
    python src/download_data.py

Dataset: Mannhardt, F. (2016). Sepsis Cases - Event Log. 4TU.ResearchData.
DOI: 10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460

If the automatic download fails (the host may change links), download
"Sepsis Cases - Event Log.xes.gz" manually from the DOI page above and place it
in data/raw/. The extracted file must be named sepsis.xes.
"""
import gzip
import shutil
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATASET_URL = (
    "https://data.4tu.nl/file/33632f3c-5c48-40cf-8d8f-2db57f5a6ce7/"
    "643dccf2-985a-459e-835c-a82bce1c0339"
)
GZ_PATH = RAW_DIR / "sepsis.xes.gz"
XES_PATH = RAW_DIR / "sepsis.xes"


def download() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if XES_PATH.exists():
        print(f"Already present: {XES_PATH}")
        return XES_PATH

    print(f"Downloading {DATASET_URL}")
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(GZ_PATH, "wb") as out:
        shutil.copyfileobj(response, out)

    print("Extracting...")
    with gzip.open(GZ_PATH, "rb") as src, open(XES_PATH, "wb") as dst:
        shutil.copyfileobj(src, dst)
    GZ_PATH.unlink()

    print(f"Saved: {XES_PATH} ({XES_PATH.stat().st_size / 1_048_576:.1f} MB)")
    return XES_PATH


if __name__ == "__main__":
    download()
