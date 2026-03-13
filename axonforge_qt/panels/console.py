"""
ConsolePanel — Bottom panel for capturing and displaying stdout/stderr output.

Features:
- QPlainTextEdit with dark theme
- Captures all stdout/stderr output via stream redirection
- Timestamps for each entry
- Auto-scroll to bottom
- 1000 lines maximum limit
- Error messages shown in red
- Toggle visibility with Shift+T shortcut
"""

from __future__ import annotations

import sys
import getpass
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QFont, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QSizePolicy
)

if TYPE_CHECKING:
    from ..bridge import BridgeAPI


class ConsoleStream:
    """Custom stream wrapper that captures output and emits to a callback."""
    
    def __init__(self, original_stream, callback, is_error: bool = False):
        self.original_stream = original_stream
        self.callback = callback
        self.is_error = is_error
        self.buffer = ""
    
    def write(self, text: str) -> None:
        """Write to both original stream and buffer for callback."""
        self.original_stream.write(text)
        self.buffer += text
        # Flush on newlines to capture complete lines
        if '\n' in text:
            self._flush_buffer()
    
    def _flush_buffer(self) -> None:
        """Flush buffered content to callback."""
        if self.buffer:
            self.callback(self.buffer, self.is_error)
            self.buffer = ""
    
    def flush(self) -> None:
        """Flush both streams."""
        self.original_stream.flush()
        self._flush_buffer()
    
    def isatty(self) -> bool:
        """Check if stream is a TTY."""
        return hasattr(self.original_stream, 'isatty') and self.original_stream.isatty()


class ConsolePanel(QWidget):
    """Bottom console panel for stdout/stderr capture and display."""
    
    # Signal for thread-safe message appending
    message_received = Signal(str, bool)
    
    def __init__(self, bridge: Optional[BridgeAPI] = None, parent=None) -> None:
        super().__init__(parent)
        self.bridge = bridge
        self.setObjectName("console_panel")
        self._disposed = False
        self._pending_blocks: list[tuple[list[str], bool]] = []
        
        # Maximum number of lines to keep
        self._max_lines = 1000
        self._line_count = 0
        
        # Store original streams
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._console_stream_out: Optional[ConsoleStream] = None
        self._console_stream_err: Optional[ConsoleStream] = None
        
        # Build UI
        self._setup_ui()
        self._apply_styling()
        
        # Connect signal for thread-safe updates
        self.message_received.connect(
            self._on_message_received,
            Qt.ConnectionType.QueuedConnection,
        )
        
        # Start capturing output
        self._start_capture()
        
        # Timer to periodically flush any remaining buffer content
        self._flush_timer = QTimer(self)
        self._flush_timer.timeout.connect(self._periodic_flush)
        self._flush_timer.start(100)  # Flush every 100ms
    
    def _setup_ui(self) -> None:
        """Set up the UI layout and widgets."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # ── Header ───────────────────────────────────────────────────────
        header = QWidget()
        header.setObjectName("console_header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 6, 10, 6)
        header_layout.setSpacing(8)
        
        # Title
        title = QLabel("Console")
        title.setObjectName("console_title")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        layout.addWidget(header)
        
        # ── Text Edit ────────────────────────────────────────────────────
        self._text_edit = QPlainTextEdit()
        self._text_edit.setObjectName("console_text_edit")
        self._text_edit.setReadOnly(True)
        self._text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Use monospace font for console output
        font = QFont("JetBrains Mono", 10)
        if not QFont(font).exactMatch():
            font = QFont("Consolas", 10)
        if not QFont(font).exactMatch():
            font = QFont("Monospace", 10)
        self._text_edit.setFont(font)
        
        layout.addWidget(self._text_edit)
        
        # Set minimum height for the panel
        self.setMinimumHeight(100)
        self.setMaximumHeight(400)
    
    def _apply_styling(self) -> None:
        """Apply custom styling to the console panel."""
        # Set the background color
        self.setStyleSheet("""
            QWidget#console_panel {
                background: #11172a;
                border-top: 1px solid #26324d;
            }
            
            QWidget#console_header {
                background: #0f1526;
                border-bottom: 1px solid #26324d;
                min-height: 32px;
                max-height: 32px;
            }
            
            QLabel#console_title {
                color: #8fa4ff;
                font-size: 0.73em;
                font-weight: 600;
                text-transform: uppercase;
                background: transparent;
            }
            
            QPlainTextEdit#console_text_edit {
                background: #11172a;
                color: #e6f1ff;
                border: none;
                padding: 8px;
                font-size: 0.8em;
                selection-background-color: #26324d;
                selection-color: #e6f1ff;
            }
            
            QPlainTextEdit#console_text_edit QScrollBar:vertical {
                background: #11172a;
                width: 8px;
            }
            
            QPlainTextEdit#console_text_edit QScrollBar::handle:vertical {
                background: #26324d;
                min-height: 20px;
            }
            
            QPlainTextEdit#console_text_edit QScrollBar::handle:vertical:hover {
                background: #8fa4ff;
            }
            
        """)
    
    def _start_capture(self) -> None:
        """Start capturing stdout and stderr."""
        # Create wrapper streams
        self._console_stream_out = ConsoleStream(
            self._original_stdout, self._on_stream_output, is_error=False
        )
        self._console_stream_err = ConsoleStream(
            self._original_stderr, self._on_stream_output, is_error=True
        )
        
        # Redirect stdout and stderr
        sys.stdout = self._console_stream_out
        sys.stderr = self._console_stream_err
    
    def _stop_capture(self) -> None:
        """Stop capturing and restore original streams."""
        if self._console_stream_out:
            self._console_stream_out.flush()
        if self._console_stream_err:
            self._console_stream_err.flush()
        
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
    
    def _on_stream_output(self, text: str, is_error: bool) -> None:
        """Handle output from the stream wrappers."""
        if self._disposed:
            return
        # Use signal for thread-safe GUI updates
        self.message_received.emit(text, is_error)
    
    def _on_message_received(self, text: str, is_error: bool) -> None:
        """Process received message (runs in GUI thread via signal)."""
        if self._disposed:
            return
        lines = text.split('\n')
        if lines and lines[-1] == "":
            lines = lines[:-1]
        if not lines or all(line.strip() == "" for line in lines):
            return
        if not self.isVisible():
            self._pending_blocks.append((lines, is_error))
            return
        self._append_message_block(lines, is_error)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._pending_blocks:
            QTimer.singleShot(0, self._flush_pending_blocks)

    def _append_message_block(self, lines: list[str], is_error: bool = False) -> None:
        """Append a multiline message block."""
        if not lines or self._disposed:
            return

        cursor = self._text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        prefix = self._build_prompt_prefix(is_error)
        continuation_prefix = " " * len(prefix)
        prefix_format = QTextCharFormat()
        if is_error:
            prefix_format.setForeground(QColor("#ff6b6b"))
        else:
            prefix_format.setForeground(QColor("#8fa4ff"))
        prefix_format.setFontWeight(QFont.Weight.DemiBold)

        for index, line in enumerate(lines):
            if index == 0:
                cursor.insertText(prefix, prefix_format)
            else:
                cursor.insertText(continuation_prefix, prefix_format)

            message_format = QTextCharFormat()
            message_format.setForeground(QColor(self._get_line_color(line, is_error)))
            cursor.insertText(line + '\n', message_format)

            self._line_count += 1
            if self._line_count > self._max_lines:
                self._trim_lines()

        self._text_edit.setTextCursor(cursor)
        self._text_edit.ensureCursorVisible()

    def _flush_pending_blocks(self) -> None:
        if self._disposed or not self.isVisible() or not self._pending_blocks:
            return
        pending = self._pending_blocks
        self._pending_blocks = []
        for lines, is_error in pending:
            self._append_message_block(lines, is_error)

    def _build_prompt_prefix(self, is_error: bool) -> str:
        """Build console prompt prefix like <user>@<project> >."""
        username = getpass.getuser() or "user"
        project = "unsaved"
        if self.bridge:
            project = str(getattr(self.bridge, "current_project_name", "unsaved") or "unsaved")
        project = project.replace(" ", "_")
        suffix = "!>" if is_error else ">"
        return f"{username}@{project} {suffix} "

    def _get_line_color(self, text: str, is_error: bool) -> str:
        """Pick a color for a single console line."""
        if is_error:
            return "#ff6b6b"

        stripped = text.strip()
        lowered = stripped.lower()

        if stripped.startswith("Traceback (most recent call last):"):
            return "#ff9f6b"
        if stripped.startswith("File \"") or stripped.startswith("at "):
            return "#f4b183"
        if "error" in lowered or lowered.endswith("exception"):
            return "#ff8a80"
        if "warning" in lowered:
            return "#ffd166"
        if lowered.startswith("ui ready") or lowered.startswith("node classes registered"):
            return "#7ec8ff"
        return "#e6f1ff"
    
    def _trim_lines(self) -> None:
        """Remove oldest lines to maintain max line limit."""
        doc = self._text_edit.document()
        block = doc.firstBlock()
        
        # Remove first block (oldest line)
        cursor = QTextCursor(block)
        cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.deleteChar()  # Remove the newline
        
        self._line_count -= 1
    
    def _periodic_flush(self) -> None:
        """Periodically flush any buffered content."""
        if self._console_stream_out:
            self._console_stream_out.flush()
        if self._console_stream_err:
            self._console_stream_err.flush()
    
    # ── Public API ─────────────────────────────────────────────────────────
    
    def append_message(self, message: str, is_error: bool = False) -> None:
        """
        Append a message to the console.
        
        Args:
            message: The message text to append
            is_error: If True, display in red color
        """
        lines = message.split('\n')
        if lines and lines[-1] == "":
            lines = lines[:-1]
        if not lines or all(line.strip() == "" for line in lines):
            return
        self._append_message_block(lines, is_error)
    
    def clear(self) -> None:
        """Clear all console output."""
        self._text_edit.clear()
        self._line_count = 0
    
    def set_visible(self, visible: bool) -> None:
        """
        Set the visibility of the entire console panel.
        
        Args:
            visible: True to show, False to hide
        """
        if visible:
            self.show()
        else:
            self.hide()
    
    def is_visible(self) -> bool:
        """Check if the console panel is visible."""
        return self.isVisible()
    
    def set_max_lines(self, max_lines: int) -> None:
        """
        Set the maximum number of lines to keep.
        
        Args:
            max_lines: Maximum number of lines (default: 1000)
        """
        self._max_lines = max(max_lines, 1)
        # Trim if needed
        while self._line_count > self._max_lines:
            self._trim_lines()
    
    def capture_exception(self, exception: Exception) -> None:
        """
        Capture and display an exception with traceback.
        
        Args:
            exception: The exception to capture
        """
        import traceback
        error_msg = f"{type(exception).__name__}: {str(exception)}"
        tb_str = traceback.format_exc()
        self.append_message(error_msg, is_error=True)
        self.append_message(tb_str, is_error=True)
    
    # ── Cleanup ─────────────────────────────────────────────────────────────
    
    def closeEvent(self, event) -> None:
        """Clean up when the panel is closed."""
        self._disposed = True
        self._pending_blocks.clear()
        self._stop_capture()
        if self._flush_timer:
            self._flush_timer.stop()
        super().closeEvent(event)
