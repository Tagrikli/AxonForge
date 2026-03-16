"""
Node branch decorator for auto-registration in the node palette.

Usage:
    from axonforge.core.descriptors import branch
    
    @branch("Input")
    class MyInputNode(Node):
        ...
    
    @branch("Input/Noise")
    class MyNoiseNode(Node):
        ...
    
    @branch("Utilities/Display")
    class MyDisplayNode(Node):
        ...
"""

from typing import Dict, List, Type, Optional
import importlib
import importlib.util
import sys
import types
from pathlib import Path


# Branch registry: {"Input": [Class1, Class2], "Input/Noise": [...], ...}
_node_branches: Dict[str, List[Type]] = {}

# Track discovered modules to avoid re-registering
_discovered_modules: set = set()


def get_node_branches() -> Dict[str, List[Type]]:
    """Get all registered node branches and their classes."""
    return _node_branches.copy()


def get_all_node_classes() -> List[Type]:
    """Get all registered node classes across all branches."""
    all_classes = []
    for classes in _node_branches.values():
        all_classes.extend(classes)
    return all_classes


def build_node_palette() -> Dict[str, List[Type]]:
    """Build the node_palette dict from registered branches."""
    return _node_branches.copy()


def clear_node_registry():
    """Clear all registered nodes (for re-discovery)."""
    global _node_branches, _discovered_modules
    _node_branches = {}
    _discovered_modules = set()


def discover_nodes(nodes_dir = None, package: Optional[str] = "axonforge.nodes") -> Dict[str, List[Type]]:
    """
    Discover all node classes in the nodes directory.

    Recursively scans Python files in the nodes directory, imports them, and registers
    any classes decorated with @branch("path"). If a Node subclass doesn't have a
    @branch decorator, it will be registered under a branch inferred from its folder path.
    For example, nodes/Utilities/Matrix/dot_product.py -> "Utilities/Matrix"

    Args:
        nodes_dir: Directory to scan (defaults to axonforge/nodes)
        package: Package name for imports. When None, uses file-based importing
                 for nodes outside the installed package (e.g. project-local nodes).

    Returns:
        Dictionary of discovered node branches
    """
    from ..node import Node  # Import here to avoid circular imports

    if nodes_dir is None:
        # Find the nodes directory relative to this file
        nodes_path: Path = Path(__file__).parent.parent.parent / "nodes"
    else:
        nodes_path = Path(nodes_dir) if isinstance(nodes_dir, str) else nodes_dir

    if not nodes_path.exists():
        return _node_branches.copy()

    # Get all Python files recursively in the nodes directory
    py_files = list(nodes_path.rglob("*.py"))

    for py_file in py_files:
        # Skip __init__.py directly in the nodes folder (but not in subdirectories)
        if py_file.name == "__init__.py" and py_file.parent == nodes_path:
            continue

        # Skip files starting with underscore (private modules), but NOT __init__.py in subdirectories
        if py_file.name.startswith("_") and py_file.name != "__init__.py":
            continue

        # Compute module path relative to nodes directory
        relative = py_file.relative_to(nodes_path)
        module_path = str(relative.with_suffix('')).replace('/', '.')

        if package is not None:
            full_module_name = f"{package}.{module_path}"
        else:
            # Synthetic module name for project-local nodes
            full_module_name = f"_axonforge_project.nodes.{module_path}"

        # Skip already discovered modules
        if full_module_name in _discovered_modules:
            continue

        try:
            if package is not None:
                # Standard package-based import
                if full_module_name in sys.modules:
                    importlib.reload(sys.modules[full_module_name])
                else:
                    importlib.import_module(full_module_name)
            else:
                # File-based import for project-local nodes
                # Register parent packages so relative imports work
                parts = full_module_name.split(".")
                for i in range(1, len(parts)):
                    parent_name = ".".join(parts[:i])
                    if parent_name not in sys.modules:
                        parent_pkg = types.ModuleType(parent_name)
                        parent_pkg.__package__ = parent_name
                        parent_pkg.__path__ = [str(nodes_path / "/".join(parts[2:i]))] if i > 2 else [str(nodes_path)]
                        sys.modules[parent_name] = parent_pkg

                spec = importlib.util.spec_from_file_location(full_module_name, py_file, submodule_search_locations=[])
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                module.__package__ = ".".join(parts[:-1])
                sys.modules[full_module_name] = module
                spec.loader.exec_module(module)

            _discovered_modules.add(full_module_name)

            # After importing, check for Node subclasses without @branch decorator
            # and register them using folder-based inference
            module = sys.modules.get(full_module_name)
            if module:
                _register_nodes_from_module(module, relative, nodes_path)

        except Exception as e:
            print(f"Warning: Failed to import {full_module_name}: {e}")

    return _node_branches.copy()


def _register_nodes_from_module(module, relative_file_path: Path, nodes_path: Path):
    """
    Scan a module for Node subclasses without @branch decorator and register them.
    
    Args:
        module: The imported Python module to scan
        relative_file_path: Path to the file relative to nodes directory
        nodes_path: The nodes directory path
    """
    from ..node import Node  # Import here to avoid circular imports
    
    # Get all classes from the module
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        
        # Check if it's a class (type) that's a subclass of Node but not Node itself
        if (isinstance(attr, type) 
            and issubclass(attr, Node) 
            and attr is not Node
            and not attr.__name__.startswith('_')):
            
            # Check if it already has a _node_branch attribute (from @branch decorator)
            if not hasattr(attr, '_node_branch'):
                # Infer branch from folder path
                branch_path = _infer_branch_from_path(relative_file_path, nodes_path)
                
                # Register under inferred branch
                if branch_path not in _node_branches:
                    _node_branches[branch_path] = []
                
                if attr not in _node_branches[branch_path]:
                    _node_branches[branch_path].append(attr)
                    attr._node_branch = branch_path


def _infer_branch_from_path(relative_file_path: Path, nodes_path: Path) -> str:
    """
    Infer the branch path from a file's folder location, capitalizing folder names
    and converting underscores to spaces.
    
    Args:
        relative_file_path: Path to the file relative to nodes directory
        nodes_path: The nodes directory path
    
    Returns:
        Branch path string with capitalized folders and spaces (e.g., "Linear Algebra/Vector")
    """
    # Get the parent directories (excluding the file itself)
    parent_parts = list(relative_file_path.parent.parts)
    
    if not parent_parts or parent_parts == ():
        # File is directly in nodes directory - use "Uncategorized"
        return "Uncategorized"
    
    # Capitalize each folder name and replace underscores with spaces
    capitalized_parts = [part.replace('_', ' ').capitalize() for part in parent_parts]
    
    # Join with forward slashes for branch path format
    branch_path = "/".join(capitalized_parts)
    return branch_path


def rediscover_nodes(nodes_dir: Optional[str] = None, package: Optional[str] = "axonforge.nodes") -> Dict[str, List[Type]]:
    """
    Re-discover all nodes by clearing registry and re-scanning.
    
    This is useful for hot-reloading new node definitions without
    restarting the server.
    
    Args:
        nodes_dir: Directory to scan (defaults to axonforge/nodes)
        package: Package name for imports
    
    Returns:
        Dictionary of discovered node branches
    """
    clear_node_registry()
    return discover_nodes(nodes_dir, package)


def branch(path: str):
    """
    Decorator for registering a node class under a branch path.
    
    The path can be a simple category like "Input" or a hierarchical
    path like "Input/Noise". The palette will display these as a
    tree structure.
    
    Usage:
        @branch("Input")
        class MyInputNode(Node):
            ...
        
        @branch("Input/Noise")
        class MyNoiseNode(Node):
            ...
    
    Args:
        path: Branch path (e.g., "Input", "Input/Noise", "Utilities/Display")
    
    Returns:
        A decorator function that registers the class under the branch
    """
    def decorator(cls: Type) -> Type:
        # Register in branch
        if path not in _node_branches:
            _node_branches[path] = []
        
        # Check if already registered (avoid duplicates on reload)
        if cls not in _node_branches[path]:
            _node_branches[path].append(cls)
        
        # Mark with branch metadata
        cls._node_branch = path
        
        return cls
    
    return decorator
