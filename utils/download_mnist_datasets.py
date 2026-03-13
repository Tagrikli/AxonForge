#!/usr/bin/env python3
"""
Download MNIST and Fashion-MNIST IDX files (gzip) for use with `python-mnist`.

Output layout:
  data/mnist/mnist/
  data/mnist/fashion-mnist/

The input nodes in this project are configured to look in these locations.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import sys
import gzip
import shutil


DATASETS = {
    "mnist": {
        "base_urls": [
            "https://storage.googleapis.com/cvdf-datasets/mnist",
        ],
        "files": [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ],
    },
    "fashion-mnist": {
        "base_urls": [
            "https://fashion-mnist.s3-website.eu-central-1.amazonaws.com",
            "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion",
            "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion",
        ],
        "files": [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ],
    },
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "MiniCortex dataset downloader"})
    with urlopen(req) as resp, dest.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 64)
            if not chunk:
                break
            f.write(chunk)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    target_root = repo_root / "data" / "mnist"
    print(f"Target root: {target_root}")

    failures = []
    for dataset_name, spec in DATASETS.items():
        dataset_dir = target_root / dataset_name
        print(f"\n[{dataset_name}] -> {dataset_dir}")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for filename in spec["files"]:
            gz_filename = filename
            uncompressed_filename = gz_filename[:-3]
            gz_dest = dataset_dir / gz_filename
            uncompressed_dest = dataset_dir / uncompressed_filename

            if uncompressed_dest.exists() and uncompressed_dest.stat().st_size > 0:
                print(f"  skip  {uncompressed_filename} (exists)")
                continue

            # Ensure gzipped file exists
            if not gz_dest.exists() or gz_dest.stat().st_size == 0:
                print(f"  fetch {gz_filename}")
                last_error = None
                for base_url in spec["base_urls"]:
                    url = f"{base_url}/{gz_filename}"
                    try:
                        download_file(url, gz_dest)
                        last_error = None
                        break
                    except (HTTPError, URLError, OSError) as exc:
                        last_error = exc
                        print(f"  fail  {gz_filename} @ {base_url}: {exc}")
                        if gz_dest.exists():
                            try:
                                gz_dest.unlink()
                            except OSError:
                                pass
                if last_error is not None:
                    failures.append((gz_filename, str(last_error)))
                    continue

            # Decompress
            print(f"  decompress {gz_filename}")
            try:
                with gzip.open(gz_dest, 'rb') as f_in:
                    with open(uncompressed_dest, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"  -> {uncompressed_filename}")
                # Clean up gz file after successful decompress
                gz_dest.unlink(missing_ok=True)
            except Exception as exc:
                print(f"  decompress fail {gz_filename}: {exc}")
                if uncompressed_dest.exists():
                    uncompressed_dest.unlink()
                failures.append((f"decompress {gz_filename}", str(exc)))

    if failures:
        print("\nSome downloads failed:", file=sys.stderr)
        for filename, err in failures:
            print(f"  - {filename}: {err}", file=sys.stderr)
        return 1

    print("\nDone.")
    print("MNIST directory: data/mnist/mnist")
    print("Fashion-MNIST directory: data/mnist/fashion-mnist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
