from __future__ import annotations

try:
    from PySide6 import QtCore, QtWidgets, QtGui  # type: ignore
except Exception:  # pragma: no cover
    try:
        from PyQt6 import QtCore, QtWidgets, QtGui  # type: ignore
    except Exception:
        from PyQt5 import QtCore, QtWidgets, QtGui  # type: ignore

# Signal compatibility
Signal = getattr(QtCore, "Signal", None)
if Signal is None:  # PyQt
    Signal = QtCore.pyqtSignal  # type: ignore

__all__ = ["QtCore", "QtWidgets", "QtGui", "Signal"]
