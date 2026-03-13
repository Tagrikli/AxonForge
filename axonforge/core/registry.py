from typing import Dict, List, Optional, TYPE_CHECKING
import threading

if TYPE_CHECKING:
    from .node import Node

_node_registry: Dict[str, "Node"] = {}
_connection_registry: List[dict] = []  # List of {from_node, from_output, to_node, to_input}
# Monotonic counter to ensure globally unique node IDs within a server run
_node_counter: int = 0
_registry_lock = threading.Lock()


def register_node(node: "Node") -> str:
    """Register a node and return its ID."""
    global _node_counter
    with _registry_lock:
        node_id = f"{node.node_type}_{_node_counter}"
        _node_counter += 1
        node.node_id = node_id
        _node_registry[node_id] = node
        return node_id


def unregister_node(node_id: str):
    """Unregister a node by ID."""
    with _registry_lock:
        if node_id in _node_registry:
            del _node_registry[node_id]


def get_node(node_id: str) -> Optional["Node"]:
    """Get a node by ID."""
    with _registry_lock:
        return _node_registry.get(node_id)


def get_all_nodes() -> Dict[str, "Node"]:
    """Get all registered nodes."""
    with _registry_lock:
        return _node_registry.copy()


def clear_node_registry():
    """Clear the node registry (for testing)."""
    global _node_registry, _connection_registry, _node_counter
    with _registry_lock:
        _node_registry = {}
        _connection_registry = []
        _node_counter = 0


def add_connection(from_node: str, from_output: str, to_node: str, to_input: str) -> bool:
    """Add a connection between nodes."""
    with _registry_lock:
        # Check if connection already exists
        for conn in _connection_registry:
            if (conn["from_node"] == from_node and
                conn["from_output"] == from_output and
                conn["to_node"] == to_node and
                conn["to_input"] == to_input):
                return False
        
        _connection_registry.append({
            "from_node": from_node,
            "from_output": from_output,
            "to_node": to_node,
            "to_input": to_input,
        })
        return True


def remove_connection(from_node: str, from_output: str, to_node: str, to_input: str) -> bool:
    """Remove a connection between nodes."""
    global _connection_registry
    with _registry_lock:
        for i, conn in enumerate(_connection_registry):
            if (conn["from_node"] == from_node and
                conn["from_output"] == from_output and
                conn["to_node"] == to_node and
                conn["to_input"] == to_input):
                _connection_registry.pop(i)
                return True
        return False


def get_connections() -> List[dict]:
    """Get all connections."""
    with _registry_lock:
        return _connection_registry.copy()


def get_connections_for_node(node_id: str) -> List[dict]:
    """Get all connections involving a specific node."""
    with _registry_lock:
        return [
            conn for conn in _connection_registry
            if conn["from_node"] == node_id or conn["to_node"] == node_id
        ]


def get_node_class(node_type: str):
    """Get a node class by type name."""
    from .descriptors.node import get_all_node_classes
    for cls in get_all_node_classes():
        if cls.__name__ == node_type:
            return cls
    return None
