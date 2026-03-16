#!/usr/bin/env python3
"""
Prefetch MNIST and Fashion-MNIST into the AxonForge cache directory.

This script downloads the raw IDX files into the user cache and leaves the
project directory untouched.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from axonforge.nodes.utilities.dataset_cache import (
    _ensure_mnist_source_available,
    _resolve_dataset_raw_dir,
)


def main() -> int:
    dataset_names = ("mnist", "fashion_mnist")
    failures = []

    for dataset_name in dataset_names:
        target_dir = _resolve_dataset_raw_dir(dataset_name)
        print(f"[{dataset_name}] -> {target_dir}")
        prepared = _ensure_mnist_source_available(dataset_name)
        if prepared is None:
            failures.append(dataset_name)
        else:
            print(f"  ready: {prepared}")

    if failures:
        print("\nSome dataset downloads failed:", file=sys.stderr)
        for name in failures:
            print(f"  - {name}", file=sys.stderr)
        return 1

    print("\nDone.")
    for dataset_name in dataset_names:
        print(f"{dataset_name}: {_resolve_dataset_raw_dir(dataset_name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
