"""
gui/widgets/monitoring_panel.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Panneau de Monitoring.
Permet d'afficher l'historique des entraînements stocké dans les fichiers CSV.
"""

import csv
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, 
    QHeaderView, QFrame
)
from PySide6.QtCore import Qt

from gui.icons import Icons, svg_icon

class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem qui permet le tri numérique s'il contient des nombres."""
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return self.text() < other.text()

# Chemins des metrics
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = PROJECT_ROOT / "monitoring" / "metrics"

class MonitoringPage(QWidget):
    """
    Page d'affichage historique des métriques FL (CSV).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pageContainer")
        self._build_ui()
        self._load_data()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # ── En-tête ────────────────────────────────────────────────────────
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)
        title = QLabel("Monitoring Historique")
        title.setObjectName("pageTitle")
        sub = QLabel("Explorez les métriques stockées dans monitoring/metrics/")
        sub.setObjectName("pageSubtitle")
        text_layout.addWidget(title)
        text_layout.addWidget(sub)
        header_layout.addLayout(text_layout)
        header_layout.addStretch()

        # Sélecteur de fichier
        self.combo_file = QComboBox()
        self.combo_file.setMinimumHeight(38)
        self.combo_file.setCursor(Qt.CursorShape.PointingHandCursor)
        self.combo_file.addItems(["Rounds (Global)", "Client Metrics (Local)", "Clients (Info)"])
        self.combo_file.currentIndexChanged.connect(self._load_data)
        self.combo_file.setStyleSheet(
            "QComboBox { padding: 4px 12px; border: 1px solid #E4E8F0; border-radius: 6px; background: white; }"
        )
        header_layout.addWidget(self.combo_file)
        
        header_layout.addSpacing(8)

        # Bouton Rafraîchir
        self.btn_refresh = QPushButton(" Rafraîchir")
        self.btn_refresh.setMinimumHeight(38)
        self.btn_refresh.setIcon(svg_icon(Icons.LOSS_CHART, 16, "#FFFFFF"))
        self.btn_refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_refresh.setStyleSheet(
            "QPushButton { padding: 0 16px; font-weight: bold; background-color: #C9956A; color: white; border-radius: 6px; } "
            "QPushButton:hover { background-color: #B5845C; }"
        )
        self.btn_refresh.clicked.connect(self._load_data)
        header_layout.addWidget(self.btn_refresh)

        layout.addLayout(header_layout)

        # ── Table de Données ───────────────────────────────────────────────
        table_container = QFrame()
        table_container.setStyleSheet(
            "QFrame { background: #FFFFFF; border: 1px solid #E4E8F0; border-radius: 12px; }"
        )
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(1, 1, 1, 1)

        self.table = QTableWidget()
        self.table.setFrameShape(QFrame.Shape.NoFrame)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setStyleSheet("""
            QTableWidget { background: transparent; gridline-color: #F0F2F7; }
            QHeaderView::section { background-color: #F8FAFC; color: #6B7280; font-weight: bold; border: none; border-right: 1px solid #F0F2F7; border-bottom: 1px solid #F0F2F7; padding: 6px; }
            QTableView::item { padding: 4px; color: #374151; }
        """)
        
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)

        table_layout.addWidget(self.table)
        layout.addWidget(table_container)

    def _load_data(self):
        """Charge le CSV correspondant au choix du ComboBox."""
        idx = self.combo_file.currentIndex()
        if idx == 0:
            filename = "rounds.csv"
        elif idx == 1:
            filename = "client_metrics.csv"
        else:
            filename = "clients.csv"
            
        filepath = METRICS_DIR / filename
        
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.table.setSortingEnabled(False)  # Désactiver le tri pendant le remplissage

        if not filepath.exists():
            self.table.setColumnCount(1)
            self.table.setHorizontalHeaderLabels(["Information"])
            self.table.insertRow(0)
            self.table.setItem(0, 0, QTableWidgetItem(f"Le fichier {filename} est introuvable. Effectuez un entrainement d'abord."))
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if not headers:
                    return

                self.table.setColumnCount(len(headers))
                self.table.setHorizontalHeaderLabels(headers)

                for row_idx, row_data in enumerate(reader):
                    self.table.insertRow(row_idx)
                    for col_idx, cell_data in enumerate(row_data):
                        item = NumericTableWidgetItem(cell_data)
                        if cell_data.replace(".", "", 1).replace("-", "", 1).isdigit():
                            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        self.table.setItem(row_idx, col_idx, item)

                for i in range(len(headers)):
                    self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
                
                self.table.setSortingEnabled(True)  # Réactiver le tri après le remplissage
                    
        except Exception as e:
            self.table.setColumnCount(1)
            self.table.setHorizontalHeaderLabels(["Erreur"])
            self.table.insertRow(0)
            self.table.setItem(0, 0, QTableWidgetItem(f"Erreur lors de la lecture du CSV : {str(e)}"))
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
