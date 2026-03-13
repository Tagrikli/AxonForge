"""
main.py - Entry point for MiniCortex Node Editor (PySide6).
"""

import faulthandler

from axonforge_qt.main import main

faulthandler.enable()
try:
    from PySide6.QtCore import qInstallMessageHandler

    def _qt_message_handler(mode, ctx, msg):
        print(f"Qt[{mode}] {msg}", flush=True)

    qInstallMessageHandler(_qt_message_handler)
except Exception:
    # Fallback if Qt isn't available yet; still keep faulthandler enabled.
    pass


if __name__ == "__main__":
    main()
