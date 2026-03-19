from typing import List, Optional, Dict, Any, Tuple, Set
import numpy as np
import traceback
import threading


class NetworkError(Exception):
    """Exception raised when a node processing error occurs."""
    def __init__(self, node_id: str, node_name: str, error: Exception, traceback_str: str):
        self.node_id = node_id
        self.node_name = node_name
        self.error = error
        self.traceback = traceback_str
        super().__init__(f"Node '{node_name}' ({node_id}) error: {error}")

from ..core.node import Node
from ..core.registry import get_node, get_connections, get_all_nodes


class Network:
    """Network manager that orchestrates computation between nodes."""
    
    def __init__(self):
        self.running = False
        self.speed: float = 60.0
        self.max_speed = False
        self.actual_hz = 0.0
        self.instant_hz = 0.0
        self._step_count = 0
        self._signals_now: Dict[Tuple[str, str], Any] = {}
        self.last_error: Optional[NetworkError] = None
        self.failed_nodes: Set[str] = set()  # Track nodes that failed processing
        self.failed_node_errors: Dict[str, str] = {}  # node_id -> latest error message
        self._step_lock = threading.Lock()
        # "Run for N steps" support
        self.run_for_target: Optional[int] = None  # target step count to stop at
        self.run_for_start: Optional[int] = None   # step count when run-for began
    
    def start(self):
        self.running = True
        self.last_error = None  # Clear error on start
    
    def stop(self):
        self.running = False
    
    def clear_failed_nodes(self) -> None:
        """Clear all tracked failed-node state."""
        self.failed_nodes.clear()
        self.failed_node_errors.clear()
        self.last_error = None
    
    def execute_step(self) -> Dict[str, Any]:
        """
        Execute one step of the network using topological ordering.
        
        Nodes are processed in dependency order so that:
        - Feedforward inputs are from the current step (fresh)
        - Feedback inputs are from the previous step (t-1)
        
        Raises NetworkError if any node fails, and stops the network.
        """
        with self._step_lock:
            nodes_by_id = get_all_nodes()
            connections = get_connections()
        
        # Build incoming port mappings and track connected nodes
        incoming: Dict[str, Dict[str, Tuple[str, str]]] = {}
        connected_nodes: Set[str] = set()
        for conn in connections:
            incoming.setdefault(conn["to_node"], {})[conn["to_input"]] = (conn["from_node"], conn["from_output"])
            connected_nodes.add(conn["from_node"])
            connected_nodes.add(conn["to_node"])

        # Get topological order (dependencies first)
        sorted_nodes = self._topological_sort(nodes_by_id, connections)

        updated_nodes = set()

        # Process nodes in topological order, skip fully disconnected nodes
        for node_id, node in sorted_nodes:
            if node_id not in connected_nodes:
                continue
            # Set input values from _signals_now
            # - Feedforward inputs: just stored by upstream nodes (current step)
            # - Feedback inputs: from previous step (cycle can't be resolved)
            self._set_node_inputs(node, incoming.get(node_id, {}))
            
            # Process the node (with error handling)
            try:
                node_outputs = self._process_node_once(node)
                # Clear this node from failed nodes if it succeeded
                self.failed_nodes.discard(node_id)
                self.failed_node_errors.pop(node_id, None)
            except Exception as e:
                # Stop the network on error
                self.running = False
                self.failed_nodes.add(node_id)  # Track the failed node
                self.failed_node_errors[node_id] = str(e)
                self.last_error = NetworkError(
                    node_id=node_id,
                    node_name=node.name,
                    error=e,
                    traceback_str=traceback.format_exc()
                )
                # Re-raise so the caller can handle it
                raise self.last_error
            
            updated_nodes.add(node_id)
            if not node_outputs:
                continue

            # Store outputs IMMEDIATELY so downstream nodes can see them this step
            for output_key, value in node_outputs.items():
                self._signals_now[(node_id, output_key)] = self._clone_signal(value)
        
        self._step_count += 1

        # Auto-stop when run-for target is reached
        if self.run_for_target is not None and self._step_count >= self.run_for_target:
            self.running = False
            self.run_for_target = None
            self.run_for_start = None

        return {"step": self._step_count, "updated_nodes": list(updated_nodes)}

    def _topological_sort(self, nodes_by_id: Dict[str, Node], connections: List[dict]) -> List[Tuple[str, Node]]:
        """
        Sort nodes so dependencies are processed before dependents.
        Uses Kahn's algorithm.
        """
        # Build dependency graph (node_id -> set of nodes it depends on)
        dependencies: Dict[str, Set[str]] = {nid: set() for nid in nodes_by_id}
        
        for conn in connections:
            # conn["to_node"] depends on conn["from_node"]
            dependencies[conn["to_node"]].add(conn["from_node"])
        
        # Kahn's algorithm
        result: List[Tuple[str, Node]] = []
        # Nodes with no dependencies are ready to process
        ready: List[str] = [nid for nid, deps in dependencies.items() if not deps]
        processed: Set[str] = set()
        
        while ready:
            # Take a node with no remaining dependencies
            node_id = ready.pop(0)
            result.append((node_id, nodes_by_id[node_id]))
            processed.add(node_id)
            
            # Remove this node from other nodes' dependencies
            for nid, deps in dependencies.items():
                if node_id in deps:
                    deps.remove(node_id)
                    # If all dependencies are satisfied, this node is ready
                    if not deps and nid not in processed:
                        ready.append(nid)
        
        # Handle any remaining nodes (cycles) - add them at the end
        for nid in nodes_by_id:
            if nid not in processed:
                result.append((nid, nodes_by_id[nid]))
        
        return result

    def propagate_current_state(self) -> Dict[str, Any]:
        nodes_by_id = get_all_nodes()
        connections = get_connections()
        self._sync_signals_from_nodes(nodes_by_id)
        
        incoming: Dict[str, Dict[str, Tuple[str, str]]] = {}
        outgoing: Dict[str, List[Tuple[str, str, str]]] = {}
        for conn in connections:
            incoming.setdefault(conn["to_node"], {})[conn["to_input"]] = (conn["from_node"], conn["from_output"])
            outgoing.setdefault(conn["from_node"], []).append((conn["to_node"], conn["to_input"], conn["from_output"]))
        
        queue: List[str] = [nid for nid in nodes_by_id if any(key[0] == nid for key in self._signals_now.keys())]
        queued = set(queue); processed = set(); updated_nodes = set()
        
        while queue:
            source_node_id = queue.pop(0)
            for (target_node_id, _, _) in outgoing.get(source_node_id, []):
                if target_node_id in processed: continue
                target_node = nodes_by_id.get(target_node_id)
                if not target_node: continue
                
                self._set_node_inputs(target_node, incoming.get(target_node_id, {}))
                try:
                    node_outputs = self._process_node_probe(target_node)
                    self.failed_nodes.discard(target_node_id)  # Clear error on success
                    self.failed_node_errors.pop(target_node_id, None)
                except Exception as e:
                    self.failed_nodes.add(target_node_id)  # Track failed node
                    self.failed_node_errors[target_node_id] = str(e)
                    self.last_error = NetworkError(
                        node_id=target_node_id,
                        node_name=target_node.name,
                        error=e,
                        traceback_str=traceback.format_exc()
                    )
                    raise self.last_error
                processed.add(target_node_id); updated_nodes.add(target_node_id)
                
                for output_key, value in node_outputs.items():
                    self._signals_now[(target_node_id, output_key)] = self._clone_signal(value)
                if target_node_id not in queued:
                    queue.append(target_node_id); queued.add(target_node_id)
        
        return {"updated_nodes": list(updated_nodes)}

    def propagate_from_node(self, node_id: str, recompute_start: bool = False) -> Dict[str, Any]:
        nodes_by_id = get_all_nodes()
        if node_id not in nodes_by_id: return {"updated_nodes": []}
        connections = get_connections()
        self._sync_signals_from_nodes(nodes_by_id)
        
        incoming: Dict[str, Dict[str, Tuple[str, str]]] = {}
        outgoing: Dict[str, List[Tuple[str, str, str]]] = {}
        for conn in connections:
            incoming.setdefault(conn["to_node"], {})[conn["to_input"]] = (conn["from_node"], conn["from_output"])
            outgoing.setdefault(conn["from_node"], []).append((conn["to_node"], conn["to_input"], conn["from_output"]))
        
        processed, updated_nodes, queue, queued = set(), set(), [], set()
        
        def enqueue(nid):
            if nid not in queued: queue.append(nid); queued.add(nid)
        
        if recompute_start:
            start_node = nodes_by_id[node_id]
            self._set_node_inputs(start_node, incoming.get(node_id, {}))
            try:
                start_outputs = self._process_node_probe(start_node)
                self.failed_nodes.discard(node_id)  # Clear error on success
                self.failed_node_errors.pop(node_id, None)
            except Exception as e:
                self.failed_nodes.add(node_id)  # Track failed node
                self.failed_node_errors[node_id] = str(e)
                self.last_error = NetworkError(
                    node_id=node_id,
                    node_name=start_node.name,
                    error=e,
                    traceback_str=traceback.format_exc()
                )
                raise self.last_error
            processed.add(node_id); updated_nodes.add(node_id)
            for output_key, value in start_outputs.items():
                self._signals_now[(node_id, output_key)] = self._clone_signal(value)
        
        enqueue(node_id)
        while queue:
            source_node_id = queue.pop(0)
            for (target_node_id, _, _) in outgoing.get(source_node_id, []):
                if target_node_id in processed: continue
                target_node = nodes_by_id.get(target_node_id)
                if not target_node: continue
                
                self._set_node_inputs(target_node, incoming.get(target_node_id, {}))
                try:
                    target_outputs = self._process_node_probe(target_node)
                    self.failed_nodes.discard(target_node_id)  # Clear error on success
                    self.failed_node_errors.pop(target_node_id, None)
                except Exception as e:
                    self.failed_nodes.add(target_node_id)  # Track failed node
                    self.failed_node_errors[target_node_id] = str(e)
                    self.last_error = NetworkError(
                        node_id=target_node_id,
                        node_name=target_node.name,
                        error=e,
                        traceback_str=traceback.format_exc()
                    )
                    raise self.last_error
                processed.add(target_node_id); updated_nodes.add(target_node_id)
                
                for output_key, value in target_outputs.items():
                    self._signals_now[(target_node_id, output_key)] = self._clone_signal(value)
                enqueue(target_node_id)
        
        return {"updated_nodes": list(updated_nodes)}
    
    def get_step_count(self) -> int: return self._step_count
    
    def reset(self):
        self._step_count = 0
        self.running = False
        self._signals_now = {}
        self.failed_nodes.clear()
        self.failed_node_errors.clear()
        self.last_error = None

    def _set_node_inputs(self, node: Node, incoming_ports: Dict[str, Tuple[str, str]]) -> None:
        """
        Set input values directly on the node instance via descriptor protocol.
        This allows nodes to access inputs via self.{port_name}.
        
        Also clears inputs that don't have connections to handle disconnections.
        """
        for port in getattr(node.__class__, "_input_ports", {}).values():
            source = incoming_ports.get(port.name)
            if source:
                v = self._signals_now.get(source)
                setattr(node, port.name, v)  # Set value (could be None if signal not yet available)
            else:
                # No connection to this port - clear any stale value
                setattr(node, port.name, None)

    def _process_node_once(self, node: Node) -> Dict[str, Any]:
        """Process a node and return its outputs."""
        if getattr(node, "_loading", False):
            return {}
        output_keys = [p.name for p in getattr(node.__class__, "_output_ports", {}).values()]
        
        # Call process() with no arguments
        node.process()
        
        # Collect outputs from OutputPort descriptors
        outputs = {}
        for key in output_keys:
            val = getattr(node, key, None)
            if val is not None:
                outputs[key] = val
        return outputs

    def _process_node_probe(self, node: Node) -> Dict[str, Any]:
        """Process a node for probing (during paused state)."""
        if getattr(node, "_loading", False):
            return {}
        output_keys = [p.name for p in getattr(node.__class__, "_output_ports", {}).values()]
        input_keys = [p.name for p in getattr(node.__class__, "_input_ports", {}).values()]
        
        # For nodes with no inputs, just process
        if not input_keys:
            node.process()
        else:
            # Check if any input has a value
            has_input = any(getattr(node, name, None) is not None for name in input_keys)
            if not has_input:
                return {}
            node.process()
        
        # Collect outputs from OutputPort descriptors
        outputs = {}
        for key in output_keys:
            val = getattr(node, key, None)
            if val is not None:
                outputs[key] = val
        return outputs

    def _sync_signals_from_nodes(self, nodes_by_id) -> None:
        """Sync signals from nodes that have output values set."""
        for node_id, node in nodes_by_id.items():
            for output_key, value in self._read_node_outputs(node).items():
                self._signals_now[(node_id, output_key)] = self._clone_signal(value)

    def _read_node_outputs(self, node: Node) -> Dict[str, Any]:
        """Read outputs from OutputPort descriptors."""
        output_keys = [p.name for p in getattr(node.__class__, "_output_ports", {}).values()]
        outputs = {}
        for key in output_keys:
            val = getattr(node, key, None)
            if val is not None:
                outputs[key] = val
        return outputs

    def _clone_signal(self, value: Any) -> Any:
        # No copying - assumes nodes don't mutate inputs in-place
        # If a node mutates shared data, the entire graph can be corrupted
        return value
