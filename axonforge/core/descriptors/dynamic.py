"""
Dynamic node decorator for hot-reload functionality.

Usage:
    @dynamic
    class MyNode(Node):
        ...
    
    # Or with a label:
    @dynamic("Experimental Node")
    class MyNode(Node):
        ...
"""

from typing import Optional, Type


def dynamic(cls_or_label: Optional[str] = None):
    """
    Decorator to mark a node class as dynamic (hot-reloadable).
    
    Can be used with or without arguments:
    
    @dynamic
    class MyNode(Node):
        ...
    
    @dynamic("Experimental")
    class MyNode(Node):
        ...
    
    When a node is marked as dynamic, it can be hot-reloaded at runtime
    by clicking the reload button (↻) in the node header. This will:
    
    - Reload the module from disk
    - Create a new instance with the updated class
    - Preserve Field values
    - Remove invalid connections (ports that no longer exist)
    - Call init() on the new instance
    
    What is PRESERVED across reload:
    - Field values (user-adjusted settings)
    - Node position
    - Node name
    - Connections to valid ports
    
    What is REINITIALIZED after reload:
    - State values (internal state variables) - NOT preserved
    - The process() method (new code)
    - InputPort/OutputPort definitions
    - Field definitions (defaults, ranges)
    - Action definitions
    - Display definitions
    - The init() method is called fresh
    """
    
    def decorator(cls: Type) -> Type:
        # Set the dynamic flag on the class
        cls.dynamic = True
        # Store optional label if provided
        if isinstance(cls_or_label, str):
            cls._dynamic_label = cls_or_label
        return cls
    
    # Handle both @dynamic and @dynamic("label") syntax
    if isinstance(cls_or_label, type):
        # @dynamic used without parentheses
        cls = cls_or_label
        cls.dynamic = True
        return cls
    else:
        # @dynamic() or @dynamic("label") used with parentheses
        return decorator
