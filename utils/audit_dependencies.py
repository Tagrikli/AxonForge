#!/usr/bin/env python3
"""Audit declared project dependencies against imports in the repository."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Dict, Iterable, List, Set


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCAN_DIRS = ("axonforge", "axonforge_qt", "utils")

DEPENDENCY_IMPORT_MAP: Dict[str, List[str]] = {
    "numpy": ["numpy"],
    "pillow": ["PIL"],
    "pyside6": ["PySide6"],
    "python-mnist": ["mnist"],
}


def _normalize_dependency_name(spec: str) -> str:
    spec = spec.strip()
    match = re.match(r"([A-Za-z0-9_.-]+)", spec)
    if not match:
        return spec.lower()
    return match.group(1).lower()


def _load_declared_dependencies(pyproject_path: Path) -> List[str]:
    data = tomllib.loads(pyproject_path.read_text())
    deps = data.get("project", {}).get("dependencies", [])
    return [_normalize_dependency_name(dep) for dep in deps]


def _iter_python_files(scan_dirs: Iterable[str]) -> Iterable[Path]:
    for name in scan_dirs:
        path = REPO_ROOT / name
        if not path.exists():
            continue
        yield from path.rglob("*.py")


def _collect_top_level_imports(file_path: Path) -> Set[str]:
    try:
        tree = ast.parse(file_path.read_text(), filename=str(file_path))
    except Exception:
        return set()

    imports: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def audit_dependencies(scan_dirs: Iterable[str]) -> Dict[str, object]:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    declared = _load_declared_dependencies(pyproject_path)

    repo_imports: Set[str] = set()
    for file_path in _iter_python_files(scan_dirs):
        repo_imports.update(_collect_top_level_imports(file_path))

    results = []
    used: List[str] = []
    unused: List[str] = []
    unknown: List[str] = []

    for dep in declared:
        import_names = DEPENDENCY_IMPORT_MAP.get(dep)
        if import_names is None:
            import_names = [dep.replace("-", "_")]
            unknown.append(dep)
        dep_used = any(name in repo_imports for name in import_names)
        results.append(
            {
                "dependency": dep,
                "imports": import_names,
                "used": dep_used,
                "mapping_known": dep in DEPENDENCY_IMPORT_MAP,
            }
        )
        if dep_used:
            used.append(dep)
        else:
            unused.append(dep)

    return {
        "scan_dirs": list(scan_dirs),
        "declared": declared,
        "used": sorted(used),
        "unused": sorted(unused),
        "unknown_mapping": sorted(unknown),
        "imports_seen": sorted(repo_imports),
        "results": results,
    }


def _print_human(result: Dict[str, object]) -> None:
    print("Dependency audit")
    print(f"Scanned: {', '.join(result['scan_dirs'])}")
    for entry in result["results"]:
        status = "used" if entry["used"] else "unused"
        mapping = "" if entry["mapping_known"] else " (mapping guessed)"
        print(
            f"- {entry['dependency']}: {status} via {', '.join(entry['imports'])}{mapping}"
        )

    unused = result["unused"]
    if unused:
        print("")
        print("Unused declared dependencies:")
        for dep in unused:
            print(f"- {dep}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit declared dependencies against imports in the repo."
    )
    parser.add_argument(
        "--scan-dir",
        action="append",
        dest="scan_dirs",
        default=None,
        help="Directory to scan for Python imports. Can be passed multiple times.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    parser.add_argument(
        "--fail-on-unused",
        action="store_true",
        help="Exit with code 1 if any declared dependency appears unused.",
    )
    args = parser.parse_args()

    scan_dirs = args.scan_dirs or list(DEFAULT_SCAN_DIRS)
    result = audit_dependencies(scan_dirs)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)

    if args.fail_on_unused and result["unused"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
