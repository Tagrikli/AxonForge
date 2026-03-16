from typing import Any, Callable, Optional, Union

import numpy as np

from .base import Display


class Numeric(Display):
    """Scalar output display descriptor."""

    def __init__(self, label: str, format: str = ".4f"):
        super().__init__(label, 0.0)
        self.format = format

    def to_spec(self, value: Any = None, obj: Any = None) -> dict:
        formatted = f"{value:{self.format}}" if isinstance(value, (int, float)) else str(value)
        return {
            "type": "numeric",
            "label": self.label,
            "format": self.format,
            "value": value,
            "formatted": formatted,
        }


class Plot(Display):
    """1D quantitative plot descriptor."""

    def __init__(
        self,
        label: str,
        style: str = "line",
        color_positive: Optional[str] = None,
        color_negative: Optional[str] = None,
        scale_mode: str = "auto",
        vmin: float = 0.0,
        vmax: float = 1.0,
        line_width: float = 1.5,
    ):
        super().__init__(label)
        self.color_positive = color_positive or "#3dff6f"
        self.color_negative = color_negative or "#e94560"
        self._default_style = style
        self._default_scale_mode = scale_mode
        self._default_vmin = float(vmin)
        self._default_vmax = float(vmax)
        self._default_line_width = float(line_width)

    def __set__(self, obj, value: Any) -> None:
        if value is not None:
            array = np.asarray(value)
            if array.ndim not in (1, 2):
                raise ValueError(
                    f"{self.name} expects a 1D or 2D array, got shape {array.shape}"
                )
        super().__set__(obj, value)

    def default_config(self) -> dict:
        return {
            "style": self._default_style,
            "scale_mode": self._default_scale_mode,
            "vmin": self._default_vmin,
            "vmax": self._default_vmax,
            "line_width": self._default_line_width,
        }

    def change_style(
        self,
        obj: Any,
        style: str,
        line_width: Optional[float] = None,
    ) -> dict:
        config = {"style": str(style)}
        if line_width is not None:
            config["line_width"] = float(line_width)
        return self.set_config(obj, config)

    def change_scale(
        self,
        obj: Any,
        scale_mode: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> dict:
        config = {"scale_mode": str(scale_mode)}
        if vmin is not None:
            config["vmin"] = float(vmin)
        if vmax is not None:
            config["vmax"] = float(vmax)
        return self.set_config(obj, config)

    def to_spec(self, value: Any = None, obj: Any = None) -> dict:
        config = self.get_config(obj) if obj is not None else self.default_config()
        return {
            "type": "plot",
            "label": self.label,
            "style": config["style"],
            "scale_mode": config["scale_mode"],
            "vmin": config["vmin"],
            "vmax": config["vmax"],
            "line_width": config["line_width"],
            "color_positive": self.color_positive,
            "color_negative": self.color_negative,
            "shape": list(value.shape) if value is not None and hasattr(value, "shape") else None,
        }


class Heatmap(Display):
    """2D scalar-field visualization descriptor."""

    def __init__(
        self,
        label: str,
        colormap: str = "grayscale",
        scale_mode: str = "auto",
        vmin: float = 0.0,
        vmax: float = 1.0,
        on_change: Union[str, Callable, None] = None,
    ):
        super().__init__(label, on_change=on_change)
        self._default_colormap = colormap
        self._default_scale_mode = scale_mode
        self._default_vmin = float(vmin)
        self._default_vmax = float(vmax)

    def default_config(self) -> dict:
        return {
            "colormap": self._default_colormap,
            "scale_mode": self._default_scale_mode,
            "vmin": self._default_vmin,
            "vmax": self._default_vmax,
        }

    def __set__(self, obj, value: Any) -> None:
        if value is not None:
            array = np.asarray(value)
            if array.ndim not in (1, 2):
                raise ValueError(
                    f"{self.name} expects a 1D or 2D array, got shape {array.shape}"
                )
        super().__set__(obj, value)

    def change_colormap(self, obj: Any, new_colormap: str) -> dict:
        return self.set_config(obj, {"colormap": new_colormap})

    def change_scale(
        self,
        obj: Any,
        scale_mode: str,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> dict:
        config = {"scale_mode": scale_mode}
        if vmin is not None:
            config["vmin"] = float(vmin)
        if vmax is not None:
            config["vmax"] = float(vmax)
        return self.set_config(obj, config)

    def to_spec(self, value: Any = None, obj: Any = None) -> dict:
        config = self.get_config(obj) if obj is not None else self.default_config()
        return {
            "type": "heatmap",
            "label": self.label,
            "colormap": config["colormap"],
            "scale_mode": config["scale_mode"],
            "vmin": config["vmin"],
            "vmax": config["vmax"],
            "shape": list(value.shape) if value is not None and hasattr(value, "shape") else None,
        }


class Image(Display):
    """2D grayscale or 3/4-channel image visualization descriptor."""

    def __set__(self, obj, value: Any) -> None:
        if value is not None:
            array = np.asarray(value)
            valid = array.ndim == 2 or (array.ndim == 3 and array.shape[2] in (3, 4))
            if not valid:
                raise ValueError(
                    f"{self.name} expects a 2D image or HxWx3/4 image, got shape {array.shape}"
                )
        super().__set__(obj, value)

    def to_spec(self, value: Any = None, obj: Any = None) -> dict:
        return {
            "type": "image",
            "label": self.label,
            "shape": list(value.shape) if value is not None and hasattr(value, "shape") else None,
        }


class Text(Display):
    """Text output display descriptor."""

    def __init__(self, label: str, default: str = "", max_height: float = 140.0):
        super().__init__(label, default)
        self.max_height = float(max_height)

    def to_spec(self, value: Any = None, obj: Any = None) -> dict:
        return {
            "type": "text",
            "label": self.label,
            "default": self.default,
            "max_height": self.max_height,
            "value": value or self.default,
        }
