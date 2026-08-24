"""
gui/widgets/nav_sidebar.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Barre de navigation latérale (dark) avec icônes SVG.
Gère la navigation entre les pages via un signal page_changed(int).
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont

from gui.icons import Icons, svg_to_pixmap, svg_icon


class NavButton(QPushButton):
    """Bouton de navigation latérale avec icône SVG et état actif."""

    # Couleurs sidebar
    COLOR_ACTIVE   = "#C9956A"
    COLOR_INACTIVE = "#7A8599"
    COLOR_BG_DARK  = "#1E2433"

    def __init__(self, icon_svg: str, label: str, page_index: int, parent=None):
        super().__init__(parent)
        self.page_index = page_index
        self._icon_svg  = icon_svg
        self._label_text = label
        self._active    = False

        self.setObjectName("navBtn")
        self.setMinimumHeight(44)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setIconSize(QSize(18, 18))
        self.setText(f"  {label}")
        self.setIcon(svg_icon(icon_svg, 18, self.COLOR_INACTIVE))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_active(self, active: bool):
        """Active ou désactive l'état sélectionné du bouton."""
        self._active = active
        color = self.COLOR_ACTIVE if active else self.COLOR_INACTIVE
        self.setObjectName("navBtnActive" if active else "navBtn")
        self.setIcon(svg_icon(self._icon_svg, 18, color))
        # Force le rechargement du QSS
        self.style().unpolish(self)
        self.style().polish(self)


class NavSidebar(QWidget):
    """
    Panneau de navigation latéral fixe (dark).

    Signal :
        page_changed (int) → index de la page à afficher
    """

    page_changed = Signal(int)

    # Définition des pages
    PAGES = [
        (Icons.DASHBOARD, "Dashboard",            0),
        (Icons.CONTROL,   "Panneau de Contrôle",  1),
        (Icons.ABOUT,     "À propos",              2),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("navSidebar")
        self.setFixedWidth(230)
        self._nav_buttons: list[NavButton] = []
        self._build_ui()
        # Active la première page par défaut
        self._select_page(0)

    # ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 20, 12, 20)
        layout.setSpacing(0)

        # ── Logo / Titre ─────────────────────────────────────────────
        logo_widget = self._build_logo()
        layout.addWidget(logo_widget)

        layout.addSpacing(24)

        # ── Séparateur ───────────────────────────────────────────────
        sep1 = self._make_separator()
        layout.addWidget(sep1)
        layout.addSpacing(16)

        # ── Label section NAVIGATION ─────────────────────────────────
        nav_label = QLabel("NAVIGATION")
        nav_label.setObjectName("navSectionLabel")
        nav_label.setContentsMargins(4, 0, 0, 0)
        layout.addWidget(nav_label)
        layout.addSpacing(8)

        # ── Boutons de navigation ────────────────────────────────────
        for icon, label, idx in self.PAGES:
            btn = NavButton(icon, label, idx)
            btn.clicked.connect(lambda checked=False, i=idx: self._select_page(i))
            self._nav_buttons.append(btn)
            layout.addWidget(btn)
            layout.addSpacing(2)

        layout.addStretch()

        # ── Séparateur bas ───────────────────────────────────────────
        sep2 = self._make_separator()
        layout.addWidget(sep2)
        layout.addSpacing(12)

        # ── Version ──────────────────────────────────────────────────
        version = QLabel("FL-IDS-OT-ICS  v1.0")
        version.setObjectName("navSectionLabel")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

    # ── Sous-widgets ─────────────────────────────────────────────────────
    def _build_logo(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 0, 0, 0)
        layout.setSpacing(10)

        # Icône settings/engrenage
        icon_lbl = QLabel()
        icon_lbl.setPixmap(svg_to_pixmap(Icons.SETTINGS, 28, "#C9956A"))
        icon_lbl.setFixedSize(32, 32)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_lbl)

        # Textes
        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        title = QLabel("FL-IDS-OT-ICS")
        title.setObjectName("navTitle")
        sub   = QLabel("FRAMEWORK")
        sub.setObjectName("navSubtitle")
        text_col.addWidget(title)
        text_col.addWidget(sub)
        layout.addLayout(text_col)
        layout.addStretch()

        return widget

    def _make_separator(self) -> QFrame:
        sep = QFrame()
        sep.setObjectName("navSeparator")
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #252D40;")
        return sep

    # ── Logique de navigation ─────────────────────────────────────────────
    def _select_page(self, index: int):
        """Met à jour l'état des boutons et émet page_changed."""
        for btn in self._nav_buttons:
            btn.set_active(btn.page_index == index)
        self.page_changed.emit(index)

    def set_active_page(self, index: int):
        """API publique pour sélectionner une page depuis l'extérieur."""
        self._select_page(index)
