from __future__ import annotations

import json
from textwrap import dedent
from typing import Any, Dict


AGENT_GUIDE = dedent(
    """
    AxonForge Agent Guide

    Core rules:
    - Subclass `Node`.
    - Put descriptors on the class body, not in `__init__()`.
    - Implement `process(self)`.
    - Assign outputs to `OutputPort` attributes. Do not return outputs from `process()`.
    - Use `State` for persisted internal values.
    - Use `Field` types only for user-editable values.
    - Use `Display` types only for visual/text output inside the node body.
    - Use `@background_init` only for slow one-time setup.

    Public authoring API:
    - Decorators: `branch`, `background_init`
    - Ports: `InputPort`, `OutputPort`
    - Fields: `Range`, `Integer`, `Float`, `Bool`, `Enum`, `DirectoryPath`, `FilePath`, `FilePaths`
    - Displays: `Numeric`, `Plot`, `Heatmap`, `Image`, `Text`
    - Other: `Action`, `State`

    Workflow:
    1. Choose the target file under a discoverable `nodes/` directory.
    2. Choose a branch with `@branch("Category/Subcategory")` if needed.
    3. Define ports, fields, displays, actions, and state on the class body.
    4. Use `init()` for setup and `process()` for runtime computation.
    5. Validate with `axonforge validate-node path/to/node.py --json`.

    Common mistakes:
    - Returning values from `process()` instead of assigning output attributes.
    - Storing persistent data in plain attributes instead of `State`.
    - Putting descriptor creation inside methods.
    - Inventing framework APIs not present in the installed package.
    """
).strip()


NODE_TEMPLATE = dedent(
    """
    import numpy as np

    from axonforge.core.node import Node
    from axonforge.core.descriptors import branch
    from axonforge.core.descriptors.ports import InputPort, OutputPort
    from axonforge.core.descriptors.fields import Float
    from axonforge.core.descriptors.displays import Text


    @branch("Utilities/Custom")
    class ExampleNode(Node):
        input_data = InputPort("Input", np.ndarray)
        output_data = OutputPort("Output", np.ndarray)

        scale = Float("Scale", default=1.0)
        info = Text("Info", default="idle")

        def process(self):
            if self.input_data is None:
                self.info = "No input"
                return

            self.output_data = self.input_data * float(self.scale)
            self.info = f"Shape: {self.input_data.shape}"
    """
).strip()


FIELDS_GUIDE = dedent(
    """
    AxonForge Field Reference

    Shared behavior:
    - Field values are stored per node instance and persisted with the graph.
    - `on_change` may be `None`, a method name string, or a callable.
    - Change callback shape:
      - string method: `def handler(self, new_value, old_value): ...`
      - callable: `def handler(obj, new_value, old_value): ...`

    `Range(label, default, min_val, max_val, step=0.01, scale="linear", on_change=None)`
    - Bounded float field.
    - `step` controls slider increment.
    - `scale` is a UI hint such as `"linear"` or `"log"`.

    `Integer(label, default=0, on_change=None)`
    - Integer field.
    - Assigned values are normalized with `int(value)`.

    `Float(label, default=0.0, on_change=None)`
    - Float field.
    - Assigned values are normalized with `float(value)`.

    `Bool(label, default=False, on_change=None)`
    - Boolean field.

    `Enum(label, options, default, on_change=None)`
    - String field restricted to `options`.

    `DirectoryPath(label, default=None, on_change=None)`
    - Stores `Path | None`.
    - Relative paths are resolved against the project root when loading.
    - Missing directories fall back to `None` on restore.

    `FilePath(label, default=None, on_change=None)`
    - Stores `Path | None`.
    - Relative paths are resolved against the project root when loading.
    - Missing files fall back to `None` on restore.

    `FilePaths(label, default=None, on_change=None)`
    - Stores `list[Path]`.
    - Missing files fall back to an empty list on restore.

    Validation command:
    - `axonforge validate-node path/to/node.py --json`
    """
).strip()


DOC_TOPICS: Dict[str, Dict[str, Any]] = {
    "agent": {
        "title": "agent",
        "content": AGENT_GUIDE,
    },
    "template": {
        "title": "template",
        "content": NODE_TEMPLATE,
    },
    "fields": {
        "title": "fields",
        "content": FIELDS_GUIDE,
    },
}


def list_doc_topics() -> list[str]:
    return sorted(DOC_TOPICS.keys())


def get_doc_topic(topic: str) -> Dict[str, Any]:
    key = topic.strip().lower()
    if key not in DOC_TOPICS:
        raise KeyError(key)
    return DOC_TOPICS[key]


def render_doc_topic(topic: str, *, as_json: bool = False) -> str:
    doc = get_doc_topic(topic)
    if as_json:
        return json.dumps(doc, indent=2)
    return str(doc["content"])
