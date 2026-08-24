"""
gui/icons/__init__.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Bibliothèque d'icônes SVG inline pour FL-IDS-OT-ICS GUI.

Toutes les icônes utilisent currentColor pour être colorisables
dynamiquement. Viewbox standard : 24×24, stroke-width : 1.5px.

Usage :
    from gui.icons import Icons, svg_icon

    btn = QPushButton("Lancer")
    btn.setIcon(svg_icon(Icons.PLAY, size=18, color="#C9956A"))
"""

from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
from PySide6.QtCore import Qt, QByteArray
from PySide6.QtSvg import QSvgRenderer


# ── Icônes SVG (viewBox="0 0 24 24", currentColor) ────────────────────────────

class Icons:
    """Constantes SVG — icônes linéaires minimalistes 24×24."""

    # ─── Navigation ─────────────────────────────────────────────────────
    DASHBOARD = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <rect x="3" y="3" width="7" height="7" rx="1.5"/>
      <rect x="14" y="3" width="7" height="7" rx="1.5"/>
      <rect x="3" y="14" width="7" height="7" rx="1.5"/>
      <rect x="14" y="14" width="7" height="7" rx="1.5"/>
    </svg>"""

    CONTROL = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <line x1="4" y1="6" x2="20" y2="6"/>
      <circle cx="8" cy="6" r="2.2" fill="currentColor" stroke="none"/>
      <line x1="4" y1="12" x2="20" y2="12"/>
      <circle cx="16" cy="12" r="2.2" fill="currentColor" stroke="none"/>
      <line x1="4" y1="18" x2="20" y2="18"/>
      <circle cx="10" cy="18" r="2.2" fill="currentColor" stroke="none"/>
    </svg>"""

    ABOUT = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="9"/>
      <circle cx="12" cy="8" r="0.8" fill="currentColor" stroke="none"/>
      <line x1="12" y1="11" x2="12" y2="17"/>
    </svg>"""

    # ─── Métriques ──────────────────────────────────────────────────────
    ROUND_COUNTER = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <polyline points="23 4 23 10 17 10"/>
      <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
    </svg>"""

    LOSS_CHART = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
    </svg>"""

    ACCURACY = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="9"/>
      <circle cx="12" cy="12" r="4"/>
      <circle cx="12" cy="12" r="0.8" fill="currentColor" stroke="none"/>
    </svg>"""

    F1_SCORE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
    </svg>"""

    # ─── Hyperparamètres ─────────────────────────────────────────────────
    ROUNDS = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <path d="M1 4v6h6"/>
      <path d="M3.51 15a9 9 0 1 0 .49-4.77"/>
    </svg>"""

    CLIENTS = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
      <circle cx="9" cy="7" r="4"/>
      <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
      <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
    </svg>"""

    EPOCHS = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <rect x="2" y="16" width="20" height="4" rx="1"/>
      <rect x="4" y="10" width="16" height="4" rx="1"/>
      <rect x="6" y="4" width="12" height="4" rx="1"/>
    </svg>"""

    LEARNING_RATE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <circle cx="8.5" cy="8.5" r="3"/>
      <circle cx="15.5" cy="15.5" r="3"/>
      <line x1="5" y1="19" x2="19" y2="5"/>
    </svg>"""

    STRATEGY = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="5" r="2.5"/>
      <circle cx="4" cy="19" r="2.5"/>
      <circle cx="20" cy="19" r="2.5"/>
      <line x1="12" y1="7.5" x2="4.8" y2="16.8"/>
      <line x1="12" y1="7.5" x2="19.2" y2="16.8"/>
      <line x1="7" y1="19" x2="17" y2="19"/>
    </svg>"""

    DATASET = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3"/>
      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
    </svg>"""

    # ─── Actions ─────────────────────────────────────────────────────────
    PLAY = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="currentColor" stroke="none">
      <polygon points="6 3 20 12 6 21 6 3"/>
    </svg>"""

    GLOBE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="9"/>
      <path d="M2.05 12h19.9"/>
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
    </svg>"""

    STOP = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="currentColor" stroke="none">
      <rect x="4" y="4" width="16" height="16" rx="2"/>
    </svg>"""

    # ─── Log Console ─────────────────────────────────────────────────────
    INFO_CIRCLE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <circle cx="12" cy="8" r="0.8" fill="currentColor" stroke="none"/>
      <line x1="12" y1="11" x2="12" y2="17"/>
    </svg>"""

    WARNING_TRIANGLE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
      <line x1="12" y1="9" x2="12" y2="13"/>
      <circle cx="12" cy="17" r="0.8" fill="currentColor" stroke="none"/>
    </svg>"""

    ERROR_CIRCLE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <line x1="15" y1="9" x2="9" y2="15"/>
      <line x1="9" y1="9" x2="15" y2="15"/>
    </svg>"""

    SUCCESS_CIRCLE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
      <polyline points="22 4 12 14.01 9 11.01"/>
    </svg>"""

    # ─── Utilitaires ─────────────────────────────────────────────────────
    TRASH = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <polyline points="3 6 5 6 21 6"/>
      <path d="M19 6l-1 14H6L5 6"/>
      <path d="M10 11v6"/>
      <path d="M14 11v6"/>
      <path d="M9 6V4h6v2"/>
    </svg>"""

    EXPORT = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>"""

    ANTENNA = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <path d="M2 16.1A5 5 0 0 1 5.9 20"/>
      <path d="M2 12.05A9 9 0 0 1 9.95 20"/>
      <path d="M2 8V6a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-6"/>
      <line x1="2" y1="20" x2="2.01" y2="20"/>
    </svg>"""

    SETTINGS = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="3"/>
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65
               1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65
               0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65
               1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0
               0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65
               1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0
               0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65
               1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0
               0 0-1.51 1z"/>
    </svg>"""

    TRENDING_DOWN = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/>
      <polyline points="17 18 23 18 23 12"/>
    </svg>"""

    BADGE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="1.6"
        stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="8" r="6"/>
      <path d="M15.477 12.89L17 22l-5-3-5 3 1.523-9.11"/>
    </svg>"""


# ── Helper : SVG string → QIcon / QPixmap ────────────────────────────────────

def svg_to_pixmap(svg_string: str, size: int = 20, color: str = "#6B7280") -> QPixmap:
    """
    Convertit une chaîne SVG en QPixmap colorisé.

    Args:
        svg_string : Chaîne SVG avec 'currentColor' comme couleur
        size       : Taille en pixels (carré)
        color      : Couleur hexadécimale pour remplacer 'currentColor'

    Returns:
        QPixmap transparent avec l'icône rendue
    """
    # Remplacement de currentColor par la couleur spécifiée
    colored_svg = svg_string.replace("currentColor", color)
    svg_bytes   = QByteArray(colored_svg.encode("utf-8"))

    renderer = QSvgRenderer(svg_bytes)
    if not renderer.isValid():
        # Fallback : pixmap vide
        px = QPixmap(size, size)
        px.fill(Qt.GlobalColor.transparent)
        return px

    px = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)

    painter = QPainter(px)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    return px


def svg_icon(svg_string: str, size: int = 20, color: str = "#6B7280") -> QIcon:
    """Retourne un QIcon à partir d'une chaîne SVG colorisée."""
    return QIcon(svg_to_pixmap(svg_string, size, color))
