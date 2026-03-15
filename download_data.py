"""Download text8 corpus and run basic sanity checks."""

import os
import zipfile
import urllib.request

DATA_DIR = "data"
URL = "http://mattmahoney.net/dc/text8.zip"


def download_text8():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "text8")

    if os.path.exists(path):
        print(f"Already exists: {path}")
    else:
        zip_path = path + ".zip"
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
        os.remove(zip_path)
        print(f"Saved to {path}")

    # Sanity checks.
    with open(path, "r") as f:
        text = f.read()

    tokens = text.split()
    unique = set(tokens)
    print(f"Tokens:      {len(tokens):,}")
    print(f"Unique:      {len(unique):,}")
    print(f"First 20:    {' '.join(tokens[:20])}")

    assert len(tokens) > 17_000_000, "corpus too small — download may be corrupted"
    assert len(unique) > 200_000, "too few unique words"


if __name__ == "__main__":
    download_text8()