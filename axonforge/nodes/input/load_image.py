"""Load an image from disk and expose it as a numpy array."""

from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from ...core.descriptors import branch
from ...core.descriptors.displays import Image
from ...core.descriptors.fields import FilePath
from ...core.descriptors.ports import OutputPort
from ...core.node import Node


@branch("Input/Image")
class LoadImage(Node):
    """Load a single image file and output it as a numpy array."""

    output_image = OutputPort("Image", np.ndarray)
    preview = Image("Image")

    file_path = FilePath(
        "File",
        on_change=lambda self, new_value, old_value: self._reload_image(force=True),
    )

    def init(self):
        self._cached_image = None
        self._cached_mtime_ns = None
        self._reload_image(force=True)

    def process(self):
        self._reload_image()
        self.output_image = self._cached_image
        self.preview = self._cached_image

    def _reload_image(self, force: bool = False) -> None:
        path = self.file_path
        if path is None:
            self._cached_image = None
            self._cached_mtime_ns = None
            self.output_image = None
            self.preview = None
            return

        image_path = Path(path)
        if not image_path.exists() or not image_path.is_file():
            self._cached_image = None
            self._cached_mtime_ns = None
            self.output_image = None
            self.preview = None
            return

        stat = image_path.stat()
        if not force and self._cached_mtime_ns == stat.st_mtime_ns:
            return

        with PILImage.open(image_path) as img:
            converted = img.convert("RGBA")
            self._cached_image = np.array(converted, dtype=np.uint8)

        self._cached_mtime_ns = stat.st_mtime_ns
        self.output_image = self._cached_image
        self.preview = self._cached_image
