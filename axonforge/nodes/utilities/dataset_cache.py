"""Dataset caching and loading helpers for built-in input nodes."""

import gzip
import os
from pathlib import Path
import shutil
import sys
import threading
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


_DATASET_LOAD_LOCK = threading.Lock()
_DATASET_CACHE = {}
_APP_NAME = "AxonForge"
_DOWNLOAD_USER_AGENT = "AxonForge dataset downloader"

_DATASETS = {
    "mnist": {
        "dir_name": "mnist",
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
    "fashion_mnist": {
        "dir_name": "fashion-mnist",
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


def _resolve_dataset_cache_dir() -> Path:
    """Resolve a persistent dataset cache directory without requiring Qt."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            base = Path(local_appdata)
        else:
            base = Path.home() / "AppData" / "Local"
    else:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache_home:
            base = Path(xdg_cache_home).expanduser()
        else:
            base = Path.home() / ".cache"

    cache_dir = base / _APP_NAME / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _resolve_dataset_raw_dir(dataset_name: str) -> Path:
    spec = _DATASETS.get(dataset_name)
    if spec is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    raw_dir = _resolve_dataset_cache_dir() / "raw" / spec["dir_name"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def _mnist_cache_paths(dataset_name: str):
    cache_dir = _resolve_dataset_cache_dir()
    images_path = cache_dir / f"{dataset_name}_train_images.npy"
    labels_path = cache_dir / f"{dataset_name}_train_labels.npy"
    return images_path, labels_path


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": _DOWNLOAD_USER_AGENT})
    with urlopen(req) as resp, dest.open("wb") as file_obj:
        while True:
            chunk = resp.read(1024 * 64)
            if not chunk:
                break
            file_obj.write(chunk)


def _ensure_mnist_source_available(dataset_name: str) -> Path | None:
    spec = _DATASETS.get(dataset_name)
    if spec is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_dir = _resolve_dataset_raw_dir(dataset_name)
    failures = []

    for gz_filename in spec["files"]:
        uncompressed_filename = gz_filename[:-3]
        gz_dest = dataset_dir / gz_filename
        uncompressed_dest = dataset_dir / uncompressed_filename

        if uncompressed_dest.exists() and uncompressed_dest.stat().st_size > 0:
            continue

        if not gz_dest.exists() or gz_dest.stat().st_size == 0:
            last_error = None
            for base_url in spec["base_urls"]:
                url = f"{base_url}/{gz_filename}"
                try:
                    _download_file(url, gz_dest)
                    last_error = None
                    break
                except (HTTPError, URLError, OSError) as exc:
                    last_error = exc
                    if gz_dest.exists():
                        try:
                            gz_dest.unlink()
                        except OSError:
                            pass
            if last_error is not None:
                failures.append(f"{gz_filename}: {last_error}")
                continue

        try:
            with gzip.open(gz_dest, "rb") as source:
                with uncompressed_dest.open("wb") as target:
                    shutil.copyfileobj(source, target)
            gz_dest.unlink(missing_ok=True)
        except Exception as exc:
            if uncompressed_dest.exists():
                uncompressed_dest.unlink()
            failures.append(f"decompress {gz_filename}: {exc}")

    if failures:
        print(
            f"Warning: failed to prepare cached dataset '{dataset_name}' in {dataset_dir}: "
            + "; ".join(failures)
        )
        return None

    return dataset_dir


def _build_mnist_cache(dataset_name: str, data_path: Path, images_path: Path, labels_path: Path) -> bool:
    """Build cached .npy files in-process."""
    try:
        from mnist import MNIST

        mn = MNIST(str(data_path))
        images, labels = mn.load_training()
        images_np = np.array(images, dtype=np.float32).reshape(-1, 28, 28) / 255.0
        labels_np = np.array(labels, dtype=np.int64)
        np.save(images_path, images_np)
        np.save(labels_path, labels_np)
        return True
    except Exception as e:
        print(f"Warning: failed to build {dataset_name} cache: {e}")
        return False


def load_dataset_with_python_mnist(dataset_name: str = "mnist"):
    """Load MNIST/Fashion-MNIST using python-mnist, backed by the user cache."""
    if dataset_name not in _DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    with _DATASET_LOAD_LOCK:
        cached = _DATASET_CACHE.get(dataset_name)
        if cached is not None:
            return cached

        images_path, labels_path = _mnist_cache_paths(dataset_name)
        if not images_path.exists() or not labels_path.exists():
            data_path = _ensure_mnist_source_available(dataset_name)
            if data_path is None:
                return None, None
            built = _build_mnist_cache(dataset_name, data_path, images_path, labels_path)
            if not built:
                return None, None

        try:
            # Use mmap for fast, low-overhead shared reads across nodes.
            images = np.load(images_path, mmap_mode="r")
            labels = np.load(labels_path, mmap_mode="r")
        except Exception as e:
            print(f"Warning: failed to load cached {dataset_name} arrays: {e}")
            return None, None

        _DATASET_CACHE[dataset_name] = (images, labels)
        return images, labels
