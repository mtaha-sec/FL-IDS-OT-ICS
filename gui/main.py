"""
gui/main.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Point d'entrée de l'interface graphique FL-IDS-OT-ICS.

Usage :
    # Depuis la racine du projet :
    python -m gui.main
    # ou :
    python gui/main.py

Prérequis :
    pip install PySide6 pyqtgraph matplotlib
"""

import sys
import os

# ── Ajout du répertoire racine au PYTHONPATH ──────────────────────────────────
# Permet d'importer les modules du projet (models/, clients/, etc.)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QFont

from gui.app import MainWindow


def main():
    """Fonction principale — initialise et lance l'application PySide6."""

    # ── Attributs de l'application (DPI, nom) ────────────────────────────
    QCoreApplication.setApplicationName("FL-IDS-OT-ICS Dashboard")
    QCoreApplication.setApplicationVersion("1.0.0")
    QCoreApplication.setOrganizationName("FL-IDS Research")

    # ── Rendu haute résolution (HiDPI) ───────────────────────────────────
    # Commentez si vous avez des problèmes d'affichage sur écran non-HiDPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)

    # ── Police par défaut ─────────────────────────────────────────────────
    font = QFont("Segoe UI", 10)
    if not font.exactMatch():
        font = QFont("Arial", 10)
    app.setFont(font)

    # ── Création et affichage de la fenêtre principale ────────────────────
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
