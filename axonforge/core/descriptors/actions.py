from typing import Any, Callable, List, Optional
from .base import BaseDescriptor


class Action(BaseDescriptor):
    """Action descriptor for triggering callbacks (e.g., button clicks)."""

    def __init__(
        self,
        label: str,
        callback: Callable[[Any, Optional[dict]], Any],
        params: Optional[List[dict]] = None,
        confirm: bool = False,
    ):
        super().__init__(label)
        self.callback = callback
        self.params = params or []
        self.confirm = confirm

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return lambda params=None: self.callback(obj, params)

    def to_spec(self, value: Any = None) -> dict:
        return {
            "type": "action",
            "label": self.label,
            "params": self.params,
            "confirm": self.confirm,
        }
