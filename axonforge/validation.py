from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple, Type

from .core.descriptors.node import _infer_branch_from_path
from .core.node import Node


PROJECT_FILE = "axonforge.json"


def _error_result(message: str, *, path: Optional[Path] = None) -> Dict[str, Any]:
    return {
        "ok": False,
        "path": str(path) if path is not None else None,
        "error": message,
        "classes": [],
    }


def _resolve_target_path(target: str) -> Path:
    path = Path(target).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_project_dir(project_dir: Optional[str]) -> Optional[Path]:
    if project_dir:
        return Path(project_dir).expanduser().resolve()

    cwd = Path.cwd().resolve()
    if (cwd / PROJECT_FILE).exists():
        return cwd
    return None


def _determine_import_mode(target_path: Path, project_dir: Optional[Path]) -> Tuple[str, str, Optional[Path], Optional[str], List[str]]:
    warnings: List[str] = []

    builtin_nodes_dir = (Path(__file__).resolve().parent / "nodes").resolve()
    try:
        builtin_relative = target_path.relative_to(builtin_nodes_dir)
    except ValueError:
        builtin_relative = None

    if builtin_relative is not None:
        module_path = str(builtin_relative.with_suffix("")).replace("/", ".")
        module_name = f"axonforge.nodes.{module_path}"
        return "builtin", module_name, builtin_nodes_dir, "axonforge.nodes", warnings

    if project_dir is not None:
        project_nodes_dir = (project_dir / "nodes").resolve()
        try:
            project_relative = target_path.relative_to(project_nodes_dir)
        except ValueError:
            project_relative = None

        if project_relative is not None:
            module_path = str(project_relative.with_suffix("")).replace("/", ".")
            module_name = f"_axonforge_project.nodes.{module_path}"
            return "project", module_name, project_nodes_dir, None, warnings

    warnings.append("Target is not under a discoverable built-in or project-local nodes directory.")
    module_name = f"_axonforge_standalone.{target_path.stem}"
    return "standalone", module_name, None, None, warnings


def _import_module_from_target(target_path: Path, module_name: str, mode: str) -> ModuleType:
    if mode == "builtin":
        if module_name in sys.modules:
            return importlib.reload(sys.modules[module_name])
        return importlib.import_module(module_name)

    spec = importlib.util.spec_from_file_location(module_name, target_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {target_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _collect_node_classes(module: ModuleType) -> List[Type[Node]]:
    classes: List[Type[Node]] = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, Node)
            and attr is not Node
            and attr.__module__ == module.__name__
            and not attr.__name__.startswith("_")
        ):
            classes.append(attr)
    return classes


def _class_summary(cls: Type[Node]) -> Dict[str, Any]:
    return {
        "name": cls.__name__,
        "branch": getattr(cls, "_node_branch", None),
        "uses_background_init": bool(getattr(cls, "_uses_background_init", False)),
        "dynamic": bool(getattr(cls, "dynamic", False)),
        "inputs": list(getattr(cls, "_input_ports", {}).keys()),
        "outputs": list(getattr(cls, "_output_ports", {}).keys()),
        "fields": list(getattr(cls, "_fields", {}).keys()),
        "actions": list(getattr(cls, "_actions", {}).keys()),
        "displays": list(getattr(cls, "_outputs", {}).keys()),
        "states": list(getattr(cls, "_states", {}).keys()),
    }


def _validate_class(
    cls: Type[Node],
    *,
    project_dir: Optional[Path],
    inferred_branch: Optional[str],
    run_init: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "name": cls.__name__,
        "ok": True,
        "warnings": [],
        "errors": [],
    }

    try:
        cls().validate_required_methods()
    except Exception as exc:
        result["ok"] = False
        result["errors"].append(f"required_methods: {exc}")
        return result

    if not hasattr(cls, "_node_branch") and inferred_branch is not None:
        result["warnings"].append(
            f"No explicit @branch decorator; inferred branch would be '{inferred_branch}'."
        )

    try:
        node = cls()
    except Exception as exc:
        result["ok"] = False
        result["errors"].append(f"instantiation: {exc}")
        return result

    try:
        schema = node.get_schema()
    except Exception as exc:
        result["ok"] = False
        result["errors"].append(f"schema: {exc}")
        return result
    else:
        result["schema"] = {
            "node_type": schema.get("node_type"),
            "category": schema.get("category"),
            "input_count": len(schema.get("input_ports", [])),
            "output_count": len(schema.get("output_ports", [])),
            "field_count": len(schema.get("fields", [])),
            "action_count": len(schema.get("actions", [])),
            "display_count": len(schema.get("outputs", [])),
            "state_count": len(schema.get("states", [])),
        }

    try:
        payload = node.to_dict(project_dir=project_dir)
        restored = cls.from_dict(payload, project_dir=project_dir)
        restored.get_schema()
    except Exception as exc:
        result["ok"] = False
        result["errors"].append(f"serialize_roundtrip: {exc}")
        return result

    if run_init:
        try:
            node.init()
        except Exception as exc:
            result["ok"] = False
            result["errors"].append(f"init: {exc}")
            result["traceback"] = traceback.format_exc()

    result["summary"] = _class_summary(cls)
    return result


def validate_node_target(
    target: str,
    *,
    class_name: Optional[str] = None,
    project_dir: Optional[str] = None,
    run_init: bool = False,
) -> Dict[str, Any]:
    target_path = _resolve_target_path(target)
    if not target_path.exists():
        return _error_result("Target file does not exist.", path=target_path)
    if target_path.suffix != ".py":
        return _error_result("Target must be a Python file.", path=target_path)

    resolved_project_dir = _resolve_project_dir(project_dir)
    mode, module_name, nodes_dir, _package, warnings = _determine_import_mode(
        target_path, resolved_project_dir
    )

    result: Dict[str, Any] = {
        "ok": True,
        "path": str(target_path),
        "mode": mode,
        "module_name": module_name,
        "project_dir": str(resolved_project_dir) if resolved_project_dir else None,
        "warnings": list(warnings),
        "classes": [],
    }

    if mode == "standalone":
        result["ok"] = False
        result["error"] = "Target is not in a discoverable nodes directory."
        return result

    try:
        module = _import_module_from_target(target_path, module_name, mode)
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"import: {exc}"
        result["traceback"] = traceback.format_exc()
        return result

    classes = _collect_node_classes(module)
    if class_name is not None:
        classes = [cls for cls in classes if cls.__name__ == class_name]

    if not classes:
        result["ok"] = False
        result["error"] = (
            f"No Node subclasses found in module matching '{class_name}'."
            if class_name
            else "No Node subclasses found in module."
        )
        return result

    inferred_branch: Optional[str] = None
    if nodes_dir is not None:
        inferred_branch = _infer_branch_from_path(target_path.relative_to(nodes_dir), nodes_dir)

    for cls in classes:
        class_result = _validate_class(
            cls,
            project_dir=resolved_project_dir,
            inferred_branch=inferred_branch,
            run_init=run_init,
        )
        result["classes"].append(class_result)

    if any(not class_result.get("ok", False) for class_result in result["classes"]):
        result["ok"] = False

    return result


def print_validation_result(result: Dict[str, Any], *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
        return

    status = "VALID" if result.get("ok") else "INVALID"
    print(f"[{status}] {result.get('path')}")
    mode = result.get("mode")
    if mode:
        print(f"Mode: {mode}")
    module_name = result.get("module_name")
    if module_name:
        print(f"Module: {module_name}")

    if result.get("project_dir"):
        print(f"Project: {result['project_dir']}")

    for warning in result.get("warnings", []):
        print(f"Warning: {warning}")

    if result.get("error"):
        print(f"Error: {result['error']}")
        if result.get("traceback"):
            print(result["traceback"])
        return

    for class_result in result.get("classes", []):
        class_status = "ok" if class_result.get("ok") else "error"
        print(f"- {class_result['name']}: {class_status}")
        summary = class_result.get("summary")
        if summary:
            print(
                "  "
                + f"branch={summary.get('branch')!r} "
                + f"inputs={len(summary.get('inputs', []))} "
                + f"outputs={len(summary.get('outputs', []))} "
                + f"fields={len(summary.get('fields', []))} "
                + f"displays={len(summary.get('displays', []))} "
                + f"states={len(summary.get('states', []))}"
            )
        for warning in class_result.get("warnings", []):
            print(f"  warning: {warning}")
        for error in class_result.get("errors", []):
            print(f"  error: {error}")

