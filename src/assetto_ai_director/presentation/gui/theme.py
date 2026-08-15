"""The dark theme, owned rather than imported.

**Why hand-written.** The obvious answer is a ready-made stylesheet package, and
the two popular ones are both dead: QDarkStyle's last release classifies up to
Python 3.10, and pyqtdarktheme pins ``<3.12`` outright. This project requires
3.12. Adopting either would mean pinning the whole application to an
end-of-life interpreter for the sake of a colour palette.

So the theme is a Fusion base with an explicit dark ``QPalette`` and a small,
object-name-scoped stylesheet. Fusion is the one Qt style that renders
identically on Windows and Linux and honours a custom palette completely — the
native Windows style ignores half of it, which is how an application ends up with
dark widgets and white scrollbars.

The palette is the load-bearing part; the stylesheet only handles the few things
a palette cannot express, such as the timeline's flat borders. Keeping it that
way means Qt's own widgets stay correct without every one of them needing a rule.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

__all__ = ["COLOURS", "apply_theme"]

COLOURS = {
    "background": QColor(24, 25, 28),
    "surface": QColor(32, 34, 38),
    "surface_high": QColor(44, 47, 52),
    "border": QColor(60, 64, 70),
    "text": QColor(226, 228, 232),
    "text_dim": QColor(140, 146, 156),
    "accent": QColor(84, 170, 255),
    "accent_dim": QColor(48, 96, 148),
    # Status. Every place these are used also carries a shape or a word, because
    # a status conveyed by colour alone is a status roughly one viewer in twelve
    # cannot read.
    "ok": QColor(104, 198, 128),
    "warning": QColor(226, 176, 84),
    "error": QColor(255, 96, 108),
    # Event and score colours, chosen to stay distinguishable for the common
    # forms of colour blindness: the pairs differ in lightness as well as hue,
    # so a viewer who cannot separate red from green still sees two shades.
    "drift": QColor(255, 138, 76),
    "corner": QColor(96, 190, 255),
    "overtake": QColor(178, 132, 255),
    "risk": QColor(255, 96, 108),
    "quiet": QColor(78, 84, 94),
}


def apply_theme(app: QApplication) -> None:
    """Apply the dark theme to an application.

    Args:
        app: The application to style.
    """
    app.setStyle("Fusion")
    app.setPalette(_palette())
    app.setStyleSheet(_stylesheet())


def _palette() -> QPalette:
    """Build the dark palette.

    Returns:
        A palette covering the active, inactive and disabled groups. Leaving the
        disabled group at its default produces unreadable grey-on-grey text,
        which is the most common way a hand-rolled dark theme goes wrong.
    """
    palette = QPalette()
    background = COLOURS["background"]
    surface = COLOURS["surface"]
    text = COLOURS["text"]
    dim = COLOURS["text_dim"]
    accent = COLOURS["accent"]

    palette.setColor(QPalette.ColorRole.Window, background)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, surface)
    palette.setColor(QPalette.ColorRole.AlternateBase, COLOURS["surface_high"])
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, surface)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, background)
    palette.setColor(QPalette.ColorRole.ToolTipBase, COLOURS["surface_high"])
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.PlaceholderText, dim)
    palette.setColor(QPalette.ColorRole.Link, accent)

    for group in (QPalette.ColorGroup.Disabled,):
        palette.setColor(group, QPalette.ColorRole.WindowText, dim)
        palette.setColor(group, QPalette.ColorRole.Text, dim)
        palette.setColor(group, QPalette.ColorRole.ButtonText, dim)

    return palette


def _stylesheet() -> str:
    """Build the small stylesheet layer.

    Returns:
        Rules for the handful of things a palette cannot express.
    """
    border = COLOURS["border"].name()
    surface = COLOURS["surface"].name()
    surface_high = COLOURS["surface_high"].name()
    accent = COLOURS["accent"].name()
    dim = COLOURS["text_dim"].name()

    return f"""
        QToolTip {{
            border: 1px solid {border};
            padding: 4px;
        }}
        QSplitter::handle {{
            background: {border};
        }}
        QSplitter::handle:horizontal {{ width: 1px; }}
        QSplitter::handle:vertical {{ height: 1px; }}
        QHeaderView::section {{
            background: {surface};
            border: none;
            border-bottom: 1px solid {border};
            padding: 6px 8px;
            color: {dim};
        }}
        QTableView {{
            gridline-color: {border};
            selection-background-color: {accent};
        }}
        QPushButton {{
            background: {surface_high};
            border: 1px solid {border};
            border-radius: 3px;
            padding: 6px 14px;
        }}
        QPushButton:hover {{ border-color: {accent}; }}
        QPushButton:disabled {{ color: {dim}; border-color: {border}; }}
        QComboBox, QSpinBox, QDoubleSpinBox {{
            background: {surface_high};
            border: 1px solid {border};
            border-radius: 3px;
            padding: 4px 8px;
        }}
        QProgressBar {{
            border: 1px solid {border};
            border-radius: 3px;
            text-align: center;
        }}
        QProgressBar::chunk {{ background: {accent}; }}
        QStatusBar {{ color: {dim}; }}
    """


def qt_alignment_centre() -> Qt.AlignmentFlag:
    """Return the centre alignment flag.

    A tiny helper so widget modules need not import ``Qt`` merely to centre a
    label.

    Returns:
        The combined centre alignment flag.
    """
    return Qt.AlignmentFlag.AlignCenter
