import argparse
import os
import zipfile
import urllib.request
from tqdm import tqdm
from .constants import OULAD_UCI_ZIP_URL, OULAD_FILES

def download(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))

def extract(zip_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

def sanity_check(out_dir: str) -> None:
    missing = [f for f in OULAD_FILES if not os.path.exists(os.path.join(out_dir, f))]
    if missing:
        raise FileNotFoundError(f"Extraction seems incomplete. Missing: {missing}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/raw/oulad", help="Where to store extracted CSVs")
    ap.add_argument("--zip_path", default="data/raw/oulad.zip", help="Where to store the downloaded zip")
    args = ap.parse_args()

    print("Downloading OULAD from UCI…")
    download(OULAD_UCI_ZIP_URL, args.zip_path)
    print("Extracting…")
    extract(args.zip_path, args.out_dir)
    sanity_check(args.out_dir)
    print(f"Done. Files extracted to: {args.out_dir}")

if __name__ == "__main__":
    main()
