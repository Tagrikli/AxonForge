from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar
import json
import pickle
import numpy as np

from .descriptors.base import Display, Field, FieldRestoreError
from .descriptors.ports import InputPort, OutputPort
from .descriptors.actions import Action
from .descriptors.state import State
from .registry import get_connections_for_node

F = TypeVar("F", bound=Callable[..., Any])


def background_init(func: F) -> F:
    """
    Mark Node.init() to run asynchronously in BridgeAPI.

    Usage:
        class MyNode(Node):
            @background_init
            def init(self):
                ...
    """
    setattr(func, "__background_init__", True)
    return func


class NodeMeta(type):
    """Metaclass that collects inputs/outputs/fields/actions from class attributes."""
    
    def __new__(mcs, name, bases, namespace):
        # Start with inherited registries so subclasses can extend/override.
        _input_ports = {}
        _output_ports = {}
        _fields = {}
        _actions = {}
        _outputs = {}
        _states = {}
        for base in reversed(bases):
            _input_ports.update(getattr(base, "_input_ports", {}))
            _output_ports.update(getattr(base, "_output_ports", {}))
            _fields.update(getattr(base, "_fields", {}))
            _actions.update(getattr(base, "_actions", {}))
            _outputs.update(getattr(base, "_outputs", {}))
            _states.update(getattr(base, "_states", {}))
        
        # Scan namespace for descriptors
        for attr_name, attr_value in namespace.items():
            if attr_name.startswith('_'):
                continue
            
            # Categories based on base classes
            if isinstance(attr_value, InputPort):
                _input_ports[attr_name] = attr_value
            elif isinstance(attr_value, OutputPort):
                _output_ports[attr_name] = attr_value
            elif isinstance(attr_value, Field):
                _fields[attr_name] = attr_value
            elif isinstance(attr_value, Action):
                _actions[attr_name] = attr_value
            elif isinstance(attr_value, Display):
                _outputs[attr_name] = attr_value
            elif isinstance(attr_value, State):
                _states[attr_name] = attr_value
        
        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Attach registries to the class
        cls._input_ports = _input_ports
        cls._output_ports = _output_ports
        cls._fields = _fields
        cls._actions = _actions
        cls._outputs = _outputs
        cls._states = _states
        init_fn = getattr(cls, "init", None)
        cls._uses_background_init = bool(
            callable(init_fn) and getattr(init_fn, "__background_init__", False)
        )
        
        return cls


class Node(metaclass=NodeMeta):
    """
    Base class for all node types.
    
    InputPort and OutputPort values are accessed directly via self.{port_name}.
    Field values are accessed directly via self.{field_name}.
    Display values are accessed directly via self.{display_name}.
    
    Example:
        class MyNode(Node):
            input_data = InputPort("Input", np.ndarray)
            output_data = OutputPort("Output", np.ndarray)
            threshold = Slider("Threshold", 0.5, 0.0, 1.0)
            result = Heatmap("Result")
            
            def process(self):
                # Access inputs directly
                if self.input_data is not None:
                    # Set outputs directly
                    self.output_data = self.input_data * self.threshold
                    self.result = self.output_data
    """
    
    # Set to True in subclasses to enable hot-reload functionality
    dynamic: bool = True
    _background_init_reserved_attrs = frozenset({
        "node_id",
        "node_type",
        "position",
        "name",
        "_output_enabled",
        "_loading",
        "_loading_error",
    })
    
    def __init__(self, x: float = 0, y: float = 0):
        self.node_id: Optional[str] = None
        # Use class name as both stable type identifier and default display name.
        self.node_type: str = self.__class__.__name__
        self.position: Dict[str, float] = {"x": x, "y": y}
        self.name: str = self.__class__.__name__
        
        # State
        self._output_enabled: Dict[str, bool] = {
            name: True for name in getattr(self.__class__, "_outputs", {}).keys()
        }
        self._loading: bool = False
        self._loading_error: str = ""

    def init(self):
        """Called once after initialization and registration."""
        pass

    def process(self):
        """
        The main compute method for the node.
        
        Access inputs via self.{input_port_name}.
        Set outputs via self.{output_port_name}.
        """
        raise NotImplementedError("Each node must implement process")

    def export_background_init_state(self) -> Dict[str, Any]:
        """
        Export node instance state produced by background init.

        The default behavior captures serializable instance attributes while
        excluding runtime identity/loading fields.
        """
        exported: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key in self._background_init_reserved_attrs:
                continue
            if callable(value):
                continue
            try:
                pickle.dumps(value)
            except Exception:
                continue
            exported[key] = value
        return exported

    def apply_background_init_state(self, state: Dict[str, Any]) -> None:
        """Apply state produced by background init worker."""
        for key, value in state.items():
            setattr(self, key, value)

    def validate_required_methods(self):
        """Validate that the node has all required methods."""
        required = ["process"]
        for method_name in required:
            if not hasattr(self, method_name):
                raise AttributeError(f"Node {self.__class__.__name__} missing required method: {method_name}")

    def get_schema(self) -> dict:
        """
        Return the node's schema for the UI.
        """
        # Input/output ports (for connections UI)
        input_ports = [p.to_spec() for p in getattr(self.__class__, "_input_ports", {}).values()]
        output_ports = [p.to_spec() for p in getattr(self.__class__, "_output_ports", {}).values()]

        # Fields (editable controls in the node body)
        fields = []
        for name, field in getattr(self.__class__, "_fields", {}).items():
            val = getattr(self, name)
            spec = field.to_spec(val)
            spec["key"] = name
            fields.append(spec)

        # Actions (buttons)
        actions = []
        for name, action in getattr(self.__class__, "_actions", {}).items():
            spec = action.to_spec()
            spec["key"] = name
            actions.append(spec)

        # Output displays (visual/text outputs shown in the node body)
        outputs = []
        for name, display in getattr(self.__class__, "_outputs", {}).items():
            val = getattr(self, name)
            spec = display.to_spec(val, obj=self)
            spec["key"] = name
            spec["enabled"] = self._output_enabled.get(name, True)
            outputs.append(spec)

        # Persisted state values (shown as informational metadata in the UI)
        states = []
        for name, state in getattr(self.__class__, "_states", {}).items():
            spec = state.to_spec()
            spec["key"] = name
            states.append(spec)

        return {
            "node_type": self.node_type,
            "node_id": self.node_id,
            "name": self.name,
            "position": self.position,
            "dynamic": getattr(self.__class__, "dynamic", False),
            "loading": self._loading,
            "loading_error": self._loading_error,
            "category": getattr(self.__class__, "_node_branch", ""),
            "input_ports": input_ports,
            "output_ports": output_ports,
            "fields": fields,
            "actions": actions,
            "outputs": outputs,
            "states": states,
        }

    def to_dict(
        self,
        array_serializer: Optional[Callable[[np.ndarray], Any]] = None,
        project_dir: Optional[Path] = None,
    ) -> dict:
        """
        Return a serializable representation of the node for persistence.
        """
        data = {
            "id": self.node_id,
            "type": self.node_type,
            "name": self.name,
            "position": self.position,
            "fields": {},
            "states": {},
        }
        
        def _serialize_value(val: Any) -> Any:
            if isinstance(val, np.ndarray):
                if array_serializer is not None:
                    return array_serializer(val)
                return {"_type": "ndarray", "data": val.tolist()}
            if isinstance(val, np.generic):
                return val.item()
            if isinstance(val, Path):
                return str(val)
            if isinstance(val, list):
                return [_serialize_value(item) for item in val]
            if isinstance(val, tuple):
                return [_serialize_value(item) for item in val]
            if isinstance(val, dict):
                return {str(key): _serialize_value(item) for key, item in val.items()}
            if isinstance(val, (int, float, str, bool, list, dict)) or val is None:
                return val
            return str(val)

        # Save all field values
        for field_name, descriptor in getattr(self.__class__, "_fields", {}).items():
            val = getattr(self, field_name)
            serialized = descriptor.serialize_value(val, project_dir=project_dir)
            data["fields"][field_name] = _serialize_value(serialized)

        # Save all state values
        for state_name in getattr(self.__class__, "_states", {}).keys():
            val = getattr(self, state_name)
            data["states"][state_name] = _serialize_value(val)
                
        return data

    @classmethod
    def from_dict(
        cls,
        data: dict,
        array_loader: Optional[Callable[[dict], Any]] = None,
        project_dir: Optional[Path] = None,
    ) -> "Node":
        """
        Create a node instance from a dictionary representation.
        """
        node = cls(
            x=data.get("position", {}).get("x", 0),
            y=data.get("position", {}).get("y", 0)
        )
        node.node_id = data.get("id")
        node.name = data.get("name", cls.__name__)
        
        _MISSING = object()

        def _deserialize_value(val: Any) -> Any:
            if isinstance(val, list):
                return [_deserialize_value(item) for item in val]
            if isinstance(val, dict):
                val_type = val.get("_type")
                if val_type == "ndarray":
                    return np.array(val["data"])
                if val_type == "ndarray_ref" and callable(array_loader):
                    try:
                        return array_loader(val)
                    except FileNotFoundError:
                        return _MISSING
                return {key: _deserialize_value(item) for key, item in val.items()}
            return val

        restore_errors = []

        # Restore field values
        for field_name, field_val in data.get("fields", {}).items():
            descriptor = getattr(cls, "_fields", {}).get(field_name)
            if not descriptor or not hasattr(node, field_name):
                continue
            try:
                deserialized = _deserialize_value(field_val)
                if deserialized is _MISSING:
                    continue
                restored = descriptor.deserialize_value(deserialized, project_dir=project_dir)
                setattr(node, field_name, restored)
            except FieldRestoreError as e:
                try:
                    setattr(node, field_name, e.fallback_value)
                except Exception:
                    pass
                restore_errors.append(str(e))
                print(f"Error restoring field {field_name}: {e}")
            except Exception as e:
                restore_errors.append(f"{descriptor.label}: {e}")
                print(f"Error restoring field {field_name}: {e}")

        # Restore state values
        for state_name, state_val in data.get("states", {}).items():
            if hasattr(node, state_name):
                try:
                    deserialized = _deserialize_value(state_val)
                    if deserialized is _MISSING:
                        continue
                    setattr(node, state_name, deserialized)
                except Exception as e:
                    print(f"Error restoring state {state_name}: {e}")
        if restore_errors:
            node._loading_error = "\n".join(restore_errors)

        return node
