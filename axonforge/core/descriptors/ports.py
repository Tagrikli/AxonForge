from typing import Any, Optional, List, TypeVar, Generic, overload

from .base import BaseDescriptor

T = TypeVar("T")


def _format_data_type(data_type: Any) -> str:
    """
    Format a data type for display.
    
    Args:
        data_type: None, str, type, or list of types
        
    Returns:
        Human-readable string representation
    """
    if data_type is None:
        return "any"
    if isinstance(data_type, str):
        return data_type
    if isinstance(data_type, (list, tuple, set)):
        return " | ".join(_format_data_type(item) for item in data_type)
    if isinstance(data_type, type):
        if data_type.__module__ == "builtins":
            return data_type.__name__
        return f"{data_type.__module__}.{data_type.__name__}"
    return str(data_type)


def _get_data_types_list(data_type: Any) -> Optional[List[str]]:
    """
    Get a list of formatted type strings for multi-type ports.
    
    Args:
        data_type: None, str, type, or list of types
        
    Returns:
        List of formatted type strings if multi-type, None otherwise
    """
    if data_type is None:
        return None
    if not isinstance(data_type, (list, tuple, set)):
        return None
    if len(data_type) <= 1:
        return None
    return [_format_data_type(item) for item in data_type]


class InputPort(BaseDescriptor, Generic[T]):
    """
    Input port descriptor - represents a connection point for incoming data.
    
    Works like Field descriptors - values are stored on the instance and
    accessed directly via self.{attribute_name}.
    
    Args:
        label: Display label for the port
        data_type: None (any type), single Type, or list of Types
        default: Default value when not connected
    
    Example:
        class MyNode(Node):
            # Accepts any type
            input_any = InputPort("Any", None)
            
            # Accepts only numpy arrays
            input_array = InputPort("Array", np.ndarray)
            
            # Accepts numpy array OR list OR tuple
            input_sequence = InputPort("Sequence", [np.ndarray, list, tuple])
    """
    
    def __init__(self, label: str, data_type: Any = None, default: Optional[T] = None):
        super().__init__(label)
        self.data_type = data_type
        self.default = default
    
    @overload
    def __get__(self, obj: None, objtype: type) -> "InputPort[T]": ...
    
    @overload
    def __get__(self, obj: object, objtype: type) -> T: ...
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, f"_{self.name}", self.default)
    
    def __set__(self, obj, value: T) -> None:
        setattr(obj, f"_{self.name}", value)
    
    def to_spec(self) -> dict:
        spec = {
            "name": self.name,
            "label": self.label,
            "data_type": _format_data_type(self.data_type),
        }
        # Include list of types for UI tooltip display
        data_types = _get_data_types_list(self.data_type)
        if data_types:
            spec["data_types"] = data_types
        return spec


class OutputPort(BaseDescriptor, Generic[T]):
    """
    Output port descriptor - represents a connection point for outgoing data.
    
    Works like Field descriptors - values are stored on the instance and
    accessed directly via self.{attribute_name}.
    
    Args:
        label: Display label for the port
        data_type: None (any type), single Type, or list of Types
        default: Default value when not set
    
    Example:
        class MyNode(Node):
            # Outputs numpy array
            output_array = OutputPort("Output", np.ndarray)
            
            # Outputs either array or list
            output_flexible = OutputPort("Flexible", [np.ndarray, list])
    """
    
    def __init__(self, label: str, data_type: Any = None, default: Optional[T] = None):
        super().__init__(label)
        self.data_type = data_type
        self.default = default
    
    @overload
    def __get__(self, obj: None, objtype: type) -> "OutputPort[T]": ...
    
    @overload
    def __get__(self, obj: object, objtype: type) -> T: ...
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, f"_{self.name}", self.default)
    
    def __set__(self, obj, value: T) -> None:
        setattr(obj, f"_{self.name}", value)
    
    def to_spec(self) -> dict:
        spec = {
            "name": self.name,
            "label": self.label,
            "data_type": _format_data_type(self.data_type),
        }
        # Include list of types for UI tooltip display
        data_types = _get_data_types_list(self.data_type)
        if data_types:
            spec["data_types"] = data_types
        return spec
