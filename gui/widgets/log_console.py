"""
gui/widgets/log_console.py  (v3 — SVG indicators, no emojis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Terminal de logs intégré. Les indicateurs de niveau utilisent
des icônes SVG colorées. Pas d'emojis.
"""

import logging
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPlainTextEdit, QPushButton, QLabel, QFrame,
)
from PySide6.QtCore import Qt, QObject, Signal, Slot
from PySide6.QtGui import QTextCharFormat, QColor, QFont, QTextCursor

from gui.icons import Icons, svg_to_pixmap


# ── Couleurs log console (fond clair) ────────────────────────────────────
LOG_COLORS = {
    "info":    "#374151",   # gris foncé    → messages normaux
    "success": "#3D8B5F",   # vert sauge    → succès / convergence
    "warning": "#B07D2A",   # ambre         → alertes
    "error":   "#C0626A",   # brique        → erreurs
    "debug":   "#9CA3AF",   # gris          → debug
    "system":  "#C9956A",   # or rose       → messages système
    "metric":  "#4A8FA8",   # teal          → métriques
}

# Mots-clés → couleur
KEYWORD_COLORS: dict[str, str] = {
    "CONVERGENCE":   "success",
    "termine":       "success",
    "terminee":      "success",
    "operationnel":  "success",
    "sauvegarde":    "success",
    "ROUND":         "system",
    "SERVEUR":       "system",
    "LOCAL TRAINER": "system",
    "Loss":          "metric",
    "Accuracy":      "metric",
    "F1":            "metric",
    "EARLY STOPPING":"warning",
    "ALERT":         "warning",
    "ERROR":         "error",
    "ARRET":         "error",
    "interrompu":    "error",
}


class QLoggingHandler(logging.Handler, QObject):
    """Intercepte les logs Python standard → signal Qt."""

    new_record = Signal(str, str)  # (message, level)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record: logging.LogRecord):
        level_map = {
            logging.DEBUG:    "debug",
            logging.INFO:     "info",
            logging.WARNING:  "warning",
            logging.ERROR:    "error",
            logging.CRITICAL: "error",
        }
        level = level_map.get(record.levelno, "info")
        self.new_record.emit(self.format(record), level)


class LogLevelIndicator(QLabel):
    """Petit QLabel avec icône SVG colorée pour le niveau de log."""

    _ICONS = {
        "info":    (Icons.INFO_CIRCLE,    "#3D8B5F"),
        "warning": (Icons.WARNING_TRIANGLE,"#B07D2A"),
        "error":   (Icons.ERROR_CIRCLE,   "#C0626A"),
    }

    def __init__(self, level: str = "info", parent=None):
        super().__init__(parent)
        icon_svg, color = self._ICONS.get(level, self._ICONS["info"])
        self.setPixmap(svg_to_pixmap(icon_svg, 12, color))
        self.setFixedSize(14, 14)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class LogConsole(QWidget):
    """
    Console de logs avec :
    - Fond sombre (contraste sur l'interface light)
    - Indicateurs de niveau SVG colorés
    - Interception des logs Python standard
    - Boutons Effacer / Exporter sans emojis
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._counts = {"info": 0, "warning": 0, "error": 0}
        self._build_ui()
        self._setup_python_logging()

    # ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Barre de titre (light mode) ─────────────────────────────────
        toolbar = QWidget()
        toolbar.setStyleSheet(
            "background: #FFFFFF;"
            "border-top: 1px solid #E4E8F0;"
            "border-bottom: 1px solid #EDF0F7;"
        )
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(16, 7, 12, 7)
        toolbar_layout.setSpacing(8)

        # Icône + titre
        icon_console = QLabel()
        icon_console.setPixmap(svg_to_pixmap(Icons.ANTENNA, 14, "#C9956A"))
        toolbar_layout.addWidget(icon_console)

        title_lbl = QLabel("Console de Logs")
        title_lbl.setStyleSheet(
            "color: #374151; font-size: 12px; font-weight: 600; "
            "background: transparent; border: none;"
        )
        toolbar_layout.addWidget(title_lbl)
        toolbar_layout.addStretch()

        # Compteurs avec icônes SVG
        for level in ["info", "warning", "error"]:
            row = QHBoxLayout()
            row.setSpacing(3)
            ind = LogLevelIndicator(level)
            row.addWidget(ind)
            names  = {"info": "Info: 0", "warning": "Warn: 0", "error": "Err: 0"}
            colors = {"info": "#3D8B5F",  "warning": "#B07D2A",  "error": "#C0626A"}
            lbl = QLabel(names[level])
            lbl.setStyleSheet(
                f"color: {colors[level]}; font-size: 11px; "
                "font-weight: 500; background: transparent; border: none;"
            )
            setattr(self, f"lbl_{level}", lbl)
            row.addWidget(lbl)
            toolbar_layout.addLayout(row)
            toolbar_layout.addSpacing(6)

        toolbar_layout.addSpacing(4)

        # Boutons Effacer / Exporter
        from PySide6.QtGui import QIcon
        btn_clear = QPushButton("  Effacer")
        btn_clear.setObjectName("btnClearLogs")
        btn_clear.setIcon(QIcon(svg_to_pixmap(Icons.TRASH, 12, "#9CA3AF")))
        btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear.clicked.connect(self.clear)
        toolbar_layout.addWidget(btn_clear)

        btn_export = QPushButton("  Exporter")
        btn_export.setObjectName("btnExport")
        btn_export.setIcon(QIcon(svg_to_pixmap(Icons.EXPORT, 12, "#9CA3AF")))
        btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_export.clicked.connect(self._export_logs)
        toolbar_layout.addWidget(btn_export)

        layout.addWidget(toolbar)

        # ── Zone de texte (fond clair) ─────────────────────────────────
        self.text_edit = QPlainTextEdit()
        self.text_edit.setObjectName("logConsole")
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text_edit.setStyleSheet(
            "QPlainTextEdit#logConsole {"
            "  background-color: #F7F8FC;"
            "  color: #374151;"
            "  border: none;"
            "  border-top: 1px solid #EDF0F7;"
            "  border-radius: 0;"
            "  font-family: 'Cascadia Code', 'Consolas', 'JetBrains Mono', monospace;"
            "  font-size: 12px;"
            "  padding: 14px 16px;"
            "  line-height: 1.7;"
            "  selection-background-color: rgba(201, 149, 106, 0.18);"
            "}"
        )

        font = QFont("Cascadia Code", 11)
        if not font.exactMatch():
            font = QFont("Consolas", 11)
        self.text_edit.setFont(font)
        layout.addWidget(self.text_edit)

    # ────────────────────────────────────────────────────────────────────
    def _setup_python_logging(self):
        """Branche QLoggingHandler sur le logger racine Python."""
        self._qt_handler = QLoggingHandler()
        self._qt_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s  [%(name)s]  %(message)s",
            datefmt="%H:%M:%S",
        ))
        self._qt_handler.new_record.connect(self._on_python_log)
        logging.getLogger().addHandler(self._qt_handler)
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Slots ─────────────────────────────────────────────────────────────
    @Slot(str)
    def append_log(self, message: str):
        """Ajoute un message depuis les signaux worker (sans emojis forcés)."""
        ts        = datetime.now().strftime("%H:%M:%S")
        color_key = self._detect_color(message)
        full_msg  = f"[{ts}]  {message}"
        self._append_colored(full_msg, color_key)

    @Slot(str, str)
    def _on_python_log(self, message: str, level: str):
        color_key = level if level in LOG_COLORS else "info"
        self._append_colored(message, color_key)
        if level in self._counts:
            self._counts[level] += 1
            names = {"info": "Info", "warning": "Warn", "error": "Err"}
            lbl = getattr(self, f"lbl_{level}", None)
            if lbl:
                lbl.setText(f"{names.get(level, level)}: {self._counts[level]}")

    def _detect_color(self, message: str) -> str:
        msg_upper = message.upper()
        for keyword, color_key in KEYWORD_COLORS.items():
            if keyword.upper() in msg_upper:
                return color_key
        return "info"

    def _append_colored(self, text: str, color_key: str):
        """Insère une ligne colorée et scrolle vers le bas."""
        color = LOG_COLORS.get(color_key, LOG_COLORS["info"])
        fmt   = QTextCharFormat()
        fmt.setForeground(QColor(color))

        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text + "\n", fmt)

        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear(self):
        """Efface le contenu et remet les compteurs à zéro."""
        self.text_edit.clear()
        self._counts = {"info": 0, "warning": 0, "error": 0}
        for level in ["info", "warning", "error"]:
            lbl = getattr(self, f"lbl_{level}", None)
            names = {"info": "Info", "warning": "Warn", "error": "Err"}
            if lbl:
                lbl.setText(f"{names.get(level, level)}: 0")

    def _export_logs(self):
        """Exporte les logs vers un fichier texte."""
        from PySide6.QtWidgets import QFileDialog
        import os
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"fl_ids_logs_{timestamp}.txt"
        default_dir  = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "monitoring", "logs"
        )
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Exporter les logs",
            os.path.join(default_dir, default_name),
            "Fichiers texte (*.txt);;Tous (*)"
        )
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.text_edit.toPlainText())
            self.append_log(f"Logs exportes vers : {filepath}")
