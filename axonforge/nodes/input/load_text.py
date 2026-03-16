"""Load text from a file and expose it as a string output."""

from pathlib import Path

from ...core.descriptors import branch
from ...core.descriptors.displays import Text
from ...core.descriptors.fields import FilePath
from ...core.descriptors.ports import OutputPort
from ...core.node import Node


@branch("Input/Text")
class LoadText(Node):
    """Load a UTF-8 text file and output its contents."""

    output_text = OutputPort("Text", str)
    preview = Text("Text", default="")

    file_path = FilePath(
        "File",
        on_change=lambda self, new_value, old_value: self._reload_text(force=True),
    )

    def init(self):
        self._cached_text = ""
        self._cached_mtime_ns = None
        self._reload_text(force=True)

    def process(self):
        self._reload_text()
        self.output_text = self._cached_text
        self.preview = self._cached_text

    def _reload_text(self, force: bool = False) -> None:
        path = self.file_path
        if path is None:
            self._cached_text = ""
            self._cached_mtime_ns = None
            self.output_text = ""
            self.preview = "No file selected"
            return

        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            self._cached_text = ""
            self._cached_mtime_ns = None
            self.output_text = ""
            self.preview = f"File not found: {file_path.name}"
            return

        stat = file_path.stat()
        if not force and self._cached_mtime_ns == stat.st_mtime_ns:
            return

        self._cached_text = file_path.read_text(encoding="utf-8", errors="replace")
        self._cached_mtime_ns = stat.st_mtime_ns
        self.output_text = self._cached_text
        self.preview = self._cached_text
