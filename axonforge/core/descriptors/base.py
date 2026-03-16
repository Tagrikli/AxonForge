from pathlib import Path
from typing import Any, Callable, Optional, Union, TypeVar, Generic, overload

T = TypeVar("T")


class BaseDescriptor:
    """Base class for all node metadata descriptors."""

    def __init__(self, label: str):
        self.label = label
        self.name: Optional[str] = None

    def __set_name__(self, owner, name: str):
        self.name = name

    def to_spec(self, value: Any = None) -> dict:
        raise NotImplementedError


class FieldRestoreError(ValueError):
    """Raised when a persisted field value cannot be restored."""

    def __init__(self, message: str, fallback_value: Any):
        super().__init__(message)
        self.fallback_value = fallback_value


class Field(BaseDescriptor, Generic[T]):
    """Base for interactive field descriptors bound to instance variables."""

    def __init__(
        self,
        label: str,
        default: T,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label)
        self.default = default
        self.on_change = on_change

    @overload
    def __get__(self, obj: None, objtype: type) -> "Field[T]": ...

    @overload
    def __get__(self, obj: object, objtype: type) -> T: ...

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, f"_{self.name}", self.default)

    def __set__(self, obj, value: T) -> None:
        old = getattr(obj, f"_{self.name}", self.default)
        normalized = self.normalize(value)
        self.validate(normalized)
        setattr(obj, f"_{self.name}", normalized)
        if normalized != old and self.on_change:
            self._trigger_change(obj, normalized, old)

    def normalize(self, value: T) -> T:
        """Override in subclasses to coerce input before validation/storage."""
        return value

    def validate(self, value: T) -> None:
        """Override in subclasses to add validation."""
        pass

    def serialize_value(self, value: T, project_dir: Optional[Path] = None) -> Any:
        """Override in subclasses to control persistence format."""
        return value

    def deserialize_value(self, value: Any, project_dir: Optional[Path] = None) -> T:
        """Override in subclasses to restore persisted values."""
        return self.normalize(value)

    def _trigger_change(self, obj, new_value: T, old_value: T) -> None:
        on_change = self.on_change
        if isinstance(on_change, str):
            callback = getattr(obj, on_change, None)
            if callable(callback):
                callback(new_value, old_value)
        elif callable(on_change):
            on_change(obj, new_value, old_value)


class Display(BaseDescriptor, Generic[T]):
    """Base for display-only output descriptors bound to instance variables."""

    def __init__(
        self,
        label: str,
        default: Optional[T] = None,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label)
        self.default = default
        self.on_change = on_change

    @overload
    def __get__(self, obj: None, objtype: type) -> "Display[T]": ...

    @overload
    def __get__(self, obj: object, objtype: type) -> T: ...

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, f"_{self.name}", self.default)

    def __set__(self, obj, value: T) -> None:
        setattr(obj, f"_{self.name}", value)

    def default_config(self) -> dict:
        return {}

    def get_config(self, obj) -> dict:
        attr_name = f"_{self.name}__display_config"
        config = getattr(obj, attr_name, None)
        if config is None:
            config = dict(self.default_config())
            setattr(obj, attr_name, config)
        return dict(config)

    def set_config(self, obj, new_config: dict) -> dict:
        attr_name = f"_{self.name}__display_config"
        old_config = self.get_config(obj)
        merged = dict(old_config)
        merged.update(new_config)
        setattr(obj, attr_name, merged)
        if merged != old_config and self.on_change:
            self._trigger_change(obj, merged, old_config)
        return merged

    def _trigger_change(self, obj, new_config: dict, old_config: dict):
        """Trigger on_change callback when display config changes."""
        on_change = self.on_change
        if isinstance(on_change, str):
            callback = getattr(obj, on_change, None)
            if callable(callback):
                callback(new_config, old_config)
        elif callable(on_change):
            on_change(obj, new_config, old_config)
