from typing import Any, Optional, TypeVar, Generic, overload

from .base import BaseDescriptor

T = TypeVar("T")


class State(BaseDescriptor, Generic[T]):
    """Descriptor for persisted internal state values."""

    def __init__(self, label: Optional[str] = None, default: Optional[T] = None):
        super().__init__(label or "")
        self.default = default

    @overload
    def __get__(self, obj: None, objtype: type) -> "State[T]": ...

    @overload
    def __get__(self, obj: object, objtype: type) -> T: ...

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, f"_{self.name}", self.default)

    def __set__(self, obj, value: T) -> None:
        setattr(obj, f"_{self.name}", value)

    def to_spec(self, value: Any = None) -> dict:
        return {
            "name": self.name,
            "label": self.label or self.name,
            "default": self.default,
        }
