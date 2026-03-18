#!/usr/bin/env python3
"""
Prefetch MNIST and Fashion-MNIST into an AxonForge project directory.

Usage:
    python utils/download_mnist_datasets.py --project-dir /path/to/project
    python utils/download_mnist_datasets.py  # uses current working directory
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from axonforge.nodes.utilities.dataset_cache import (
    set_project_dir,
    _ensure_mnist_source_available,
    _resolve_dataset_raw_dir,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefetch MNIST datasets into a project directory.")
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

    set_project_dir(project_dir)

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
