from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Union

from .base import Field, FieldRestoreError


def _serialize_project_path(path: Path, project_dir: Optional[Path]) -> str:
    path_obj = Path(path).expanduser()
    if project_dir is not None:
        project_root = Path(project_dir).expanduser().resolve()
        if not path_obj.is_absolute():
            path_obj = project_root / path_obj
        try:
            return path_obj.resolve(strict=False).relative_to(project_root).as_posix() or "."
        except ValueError:
            pass
    if path_obj.is_absolute():
        return str(path_obj.resolve(strict=False))
    return path_obj.as_posix()


def _resolve_loaded_path(value: Union[str, Path], project_dir: Optional[Path]) -> Path:
    path_obj = Path(value).expanduser()
    if not path_obj.is_absolute() and project_dir is not None:
        return (Path(project_dir).expanduser().resolve() / path_obj).resolve(strict=False)
    if path_obj.is_absolute():
        return path_obj.resolve(strict=False)
    return path_obj


class Range(Field[float]):
    """Numeric range field descriptor with min/max bounds."""

    def __init__(
        self,
        label: str,
        default: float,
        min_val: float,
        max_val: float,
        step: float = 0.01,
        scale: str = "linear",
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label, default, on_change)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.scale = scale

    def validate(self, value: float) -> None:
        if not (self.min_val <= value <= self.max_val):
            raise ValueError(
                f"{self.name} must be in [{self.min_val}, {self.max_val}], got {value}"
            )

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "range",
            "label": self.label,
            "default": self.default,
            "value": value if value is not None else self.default,
            "min": self.min_val,
            "max": self.max_val,
            "step": self.step,
            "scale": self.scale,
        }


class Integer(Field[int]):
    """Integer field descriptor."""

    def __init__(
        self,
        label: str,
        default: int = 0,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label, int(default), on_change)

    def normalize(self, value: Any) -> int:
        return int(value)

    def validate(self, value: int) -> None:
        try:
            int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{self.name} must be an integer, got {value}")

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "integer",
            "label": self.label,
            "default": self.default,
            "value": value if value is not None else self.default,
        }


class Float(Field[float]):
    """Float field descriptor."""

    def __init__(
        self,
        label: str,
        default: float = 0.0,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label, float(default), on_change)

    def normalize(self, value: Any) -> float:
        return float(value)

    def validate(self, value: float) -> None:
        try:
            float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{self.name} must be a float, got {value}")

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "float",
            "label": self.label,
            "default": self.default,
            "value": value if value is not None else self.default,
        }


class Bool(Field[bool]):
    """Boolean field descriptor."""

    def __init__(
        self,
        label: str,
        default: bool = False,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label, default, on_change)

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "bool",
            "label": self.label,
            "default": self.default,
            "value": value if value is not None else self.default,
        }


class Enum(Field[str]):
    """Enum selection field descriptor."""

    def __init__(
        self,
        label: str,
        options: List[str],
        default: str,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label, default, on_change)
        self.options = options

    def normalize(self, value: Any) -> str:
        return str(value)

    def validate(self, value: str) -> None:
        if value not in self.options:
            raise ValueError(f"{self.name} must be one of {self.options}, got {value}")

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "enum",
            "label": self.label,
            "options": self.options,
            "default": self.default,
            "value": value if value is not None else self.default,
        }


class DirectoryPath(Field[Optional[Path]]):
    """Directory picker field that stores a single Path at runtime."""

    def __init__(
        self,
        label: str,
        default: Optional[Union[str, Path]] = None,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label, self.normalize(default), on_change)

    def normalize(self, value: Any) -> Optional[Path]:
        if value is None or value == "":
            return None
        if isinstance(value, Path):
            return value.expanduser()
        return Path(str(value)).expanduser()

    def validate(self, value: Optional[Path]) -> None:
        if value is not None and not isinstance(value, Path):
            raise ValueError(f"{self.name} must be a Path or None, got {value!r}")

    def serialize_value(self, value: Optional[Path], project_dir: Optional[Path] = None) -> Any:
        if value is None:
            return None
        return _serialize_project_path(value, project_dir)

    def deserialize_value(self, value: Any, project_dir: Optional[Path] = None) -> Optional[Path]:
        normalized = self.normalize(value)
        if normalized is None:
            return None
        resolved = _resolve_loaded_path(normalized, project_dir)
        if not resolved.exists() or not resolved.is_dir():
            raise FieldRestoreError(
                f"{self.label}: directory not found: {resolved}",
                None,
            )
        return resolved

    def to_spec(self, value: Any = None) -> dict:
        current = value if value is not None else self.default
        return {
            "type": "directory_path",
            "label": self.label,
            "default": str(self.default) if self.default is not None else "",
            "value": str(current) if current is not None else "",
        }


class FilePath(Field[Optional[Path]]):
    """File picker field that stores a single Path at runtime."""

    def __init__(
        self,
        label: str,
        default: Optional[Union[str, Path]] = None,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label, self.normalize(default), on_change)

    def normalize(self, value: Any) -> Optional[Path]:
        if value is None or value == "":
            return None
        if isinstance(value, Path):
            return value.expanduser()
        return Path(str(value)).expanduser()

    def validate(self, value: Optional[Path]) -> None:
        if value is not None and not isinstance(value, Path):
            raise ValueError(f"{self.name} must be a Path or None, got {value!r}")

    def serialize_value(self, value: Optional[Path], project_dir: Optional[Path] = None) -> Any:
        if value is None:
            return None
        return _serialize_project_path(value, project_dir)

    def deserialize_value(self, value: Any, project_dir: Optional[Path] = None) -> Optional[Path]:
        normalized = self.normalize(value)
        if normalized is None:
            return None
        resolved = _resolve_loaded_path(normalized, project_dir)
        if not resolved.exists() or not resolved.is_file():
            raise FieldRestoreError(
                f"{self.label}: file not found: {resolved}",
                None,
            )
        return resolved

    def to_spec(self, value: Any = None) -> dict:
        current = value if value is not None else self.default
        return {
            "type": "file_path",
            "label": self.label,
            "default": str(self.default) if self.default is not None else "",
            "value": str(current) if current is not None else "",
        }


class FilePaths(Field[List[Path]]):
    """File picker field that stores multiple Paths at runtime."""

    def __init__(
        self,
        label: str,
        default: Optional[Iterable[Union[str, Path]]] = None,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label, self.normalize(default), on_change)

    def normalize(self, value: Any) -> List[Path]:
        if value is None or value == "":
            return []
        if isinstance(value, (str, Path)):
            return [Path(value).expanduser()]
        if isinstance(value, Iterable):
            result: List[Path] = []
            for item in value:
                if isinstance(item, Path):
                    result.append(item.expanduser())
                else:
                    result.append(Path(str(item)).expanduser())
            return result
        raise ValueError(f"{self.name} must be an iterable of paths, got {value!r}")

    def validate(self, value: List[Path]) -> None:
        if not isinstance(value, list):
            raise ValueError(f"{self.name} must be a list of Path objects, got {value!r}")
        for item in value:
            if not isinstance(item, Path):
                raise ValueError(f"{self.name} must contain only Path objects, got {item!r}")

    def serialize_value(self, value: List[Path], project_dir: Optional[Path] = None) -> Any:
        return [_serialize_project_path(item, project_dir) for item in value]

    def deserialize_value(self, value: Any, project_dir: Optional[Path] = None) -> List[Path]:
        resolved_paths = [_resolve_loaded_path(item, project_dir) for item in self.normalize(value)]
        missing = [path for path in resolved_paths if not path.exists() or not path.is_file()]
        if missing:
            preview = ", ".join(str(path) for path in missing[:3])
            suffix = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
            raise FieldRestoreError(
                f"{self.label}: file paths not found: {preview}{suffix}",
                [],
            )
        return resolved_paths

    def to_spec(self, value: Any = None) -> dict:
        current = value if value is not None else self.default
        values = [str(item) for item in current]
        return {
            "type": "file_paths",
            "label": self.label,
            "default": [str(item) for item in self.default],
            "value": values,
        }
