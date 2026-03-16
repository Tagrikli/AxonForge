"""
main.py — Entry point for the PySide6 AxonForge Node Editor.
"""

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QApplication

from .bridge import BridgeAPI
from .main_window import MainWindow


def load_stylesheet() -> str:
    """Load the QSS theme file."""
    qss_path = Path(__file__).parent / "themes" / "dark.qss"
    if qss_path.exists():
        return qss_path.read_text()
    return ""


def main(project_dir: Path, graph_name: Optional[str] = None) -> None:
    print("AxonForge — PySide6 Node Editor")
    print("=" * 40)

    app = QApplication(sys.argv)

    # Load stylesheet
    qss = load_stylesheet()
    if qss:
        app.setStyleSheet(qss)

    # Initialise bridge (discovers nodes, builds palette)
    bridge = BridgeAPI(project_dir=project_dir)
    bridge.init()

    # Load initial graph if specified
    if graph_name:
        bridge.load_graph(graph_name)

    # Create and show main window
    window = MainWindow(bridge, ui_fps=60.0)
    window.show()

    print(f"AxonForge ready. Project: {bridge.project_name}")
    sys.exit(app.exec())
