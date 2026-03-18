#!/usr/bin/env python3
"""
Download CIFAR-10 dataset into an AxonForge project directory.

Usage:
    python utils/download_cifar10.py --project-dir /path/to/project
    python utils/download_cifar10.py  # uses current working directory

Output layout:
  ax_data/cifar10/
    data_batch_1 .. data_batch_5
    test_batch
    batches.meta
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import tarfile
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from axonforge.project import DIR_DATA


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def download_file(url: str, dest: Path) -> None:
    """Download a file from URL to destination."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "MiniCortex dataset downloader"})
    
    print(f"Downloading from {url}...")
    with urlopen(req) as resp, dest.open("wb") as f:
        total_size = int(resp.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 1024 * 64
        
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\r  Progress: {percent:.1f}%", end="", flush=True)
        print()  # New line after progress


def extract_tar_gz(tar_path: Path, extract_to: Path) -> None:
    """Extract tar.gz file to destination directory."""
    print(f"Extracting {tar_path.name}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    
    print(f"Extracted to {extract_to}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CIFAR-10 into an AxonForge project directory.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Path to the AxonForge project directory (default: current working directory)",
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    if not (project_dir / "axonforge.json").exists():
        print(f"Error: {project_dir} does not look like an AxonForge project (axonforge.json not found).", file=sys.stderr)
        return 1

    data_dir = project_dir / DIR_DATA
    cifar_dir = data_dir / "cifar10"
    temp_dir = data_dir / "temp"
    
    print(f"Target directory: {cifar_dir}")
    
    # Check if CIFAR-10 is already downloaded
    if cifar_dir.exists() and any(cifar_dir.iterdir()):
        print("CIFAR-10 appears to already be downloaded.")
        response = input("Re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            return 0
    
    # Download the tar.gz file
    tar_path = temp_dir / "cifar-10-python.tar.gz"
    try:
        download_file(CIFAR10_URL, tar_path)
    except (HTTPError, URLError, OSError) as exc:
        print(f"Failed to download CIFAR-10: {exc}", file=sys.stderr)
        return 1
    
    # Extract the tar.gz file
    try:
        extract_tar_gz(tar_path, data_dir)
    except tarfile.TarError as exc:
        print(f"Failed to extract CIFAR-10: {exc}", file=sys.stderr)
        return 1
    
    # Move files from cifar-10-batches-py to cifar10
    extracted_dir = data_dir / "cifar-10-batches-py"
    if extracted_dir.exists():
        cifar_dir.mkdir(parents=True, exist_ok=True)
        
        # Move all batch files
        for batch_file in extracted_dir.glob("*"):
            dest = cifar_dir / batch_file.name
            batch_file.rename(dest)
            print(f"  Moved {batch_file.name} -> {cifar_dir}")
        
        # Remove empty directory
        extracted_dir.rmdir()
        print(f"Cleaned up {extracted_dir}")
    
    # Clean up temp file
    if tar_path.exists():
        tar_path.unlink()
        print(f"Removed {tar_path}")
    
    # Clean up temp directory
    if temp_dir.exists():
        temp_dir.rmdir()
        print(f"Removed {temp_dir}")
    
    print("\nDone!")
    print(f"CIFAR-10 dataset is ready at: {cifar_dir}")
    print("\nDataset contents:")
    for f in sorted(cifar_dir.glob("*")):
        print(f"  - {f.name}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
