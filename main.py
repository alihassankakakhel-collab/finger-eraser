"""Launch the finger-controlled camera eraser."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORMTHEME", "generic")

import cv2

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_FONTDIR", None)

from PySide6.QtWidgets import QApplication

from .live_overlay import LiveEraserWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Finger Eraser")
    app.setStyle("Fusion")
    window = LiveEraserWindow()
    window.show()
    return app.exec()
