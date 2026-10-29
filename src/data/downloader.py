"""
Downloads the Salminen et al. (2022) Fake Reviews Dataset from OSF.
Dataset: https://osf.io/tyue9/
"""
import os
import requests
from pathlib import Path
from tqdm import tqdm

OSF_DOWNLOAD_URL = "https://osf.io/tyue9/download"
DEFAULT_DEST = Path("data/raw/fake_reviews_dataset.csv")

def download_salminen_dataset(dest: Path = DEFAULT_DEST, overwrite: bool = False) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        print(f"Dataset already exists at {dest}. Skipping download.")
        return dest

    print(f"Downloading Salminen et al. (2022) dataset to {dest} ...")
    response = requests.get(OSF_DOWNLOAD_URL, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"Download complete: {dest}")
    return dest


if __name__ == "__main__":
    download_salminen_dataset()
