"""
gui/app.py  (v3 — Navigation par pages + Light Mode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fenêtre principale avec :
  - NavSidebar (dark) → navigation entre 3 pages
  - QStackedWidget :
      Page 0 : Dashboard (ChartsPanel + LogConsole en bas)
      Page 1 : Panneau de Contrôle (ControlPanel)
      Page 2 : A propos du Framework
  - Thème : light_theme.qss
"""

import os
import sys
from datetime import datetime

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QApplication,
    QVBoxLayout, QHBoxLayout, QSplitter, QSplitterHandle,
    QStackedWidget, QLabel, QFrame, QScrollArea,
)
from PySide6.QtCore import Qt, QSize, Slot, QTimer, QRect
from PySide6.QtGui import QFont, QColor, QPainter, QBrush, QPen

from gui.widgets.nav_sidebar  import NavSidebar
from gui.widgets.control_panel import ControlPanel
from gui.widgets.log_console   import LogConsole
from gui.widgets.charts_panel  import ChartsPanel
from gui.icons                 import Icons, svg_to_pixmap, svg_icon

from gui.workers.fl_runner import FLRunnerWorker

# ── Feuille de style light ────────────────────────────────────────────────────
_QSS_PATH = os.path.join(
    os.path.dirname(__file__), "styles", "light_theme.qss"
)


def _load_qss(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "QMainWindow { background: #F4F6FA; color: #1A1D2E; }"


# ────────────────────────────────────────────────────────────────────────────
# Page "A propos du Framework"
# ────────────────────────────────────────────────────────────────────────────
class AboutPage(QWidget):
    """Page d'information sur le framework FL-IDS-OT-ICS."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pageContainer")
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: #F4F6FA; border: none; }")

        content = QWidget()
        content.setStyleSheet("background: #F4F6FA;")
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        # En-tête
        title = QLabel("A propos du Framework")
        title.setObjectName("pageTitle")
        sub   = QLabel("Architecture, données et points d'intégration FL-IDS-OT-ICS")
        sub.setObjectName("pageSubtitle")
        main_layout.addWidget(title)
        main_layout.addWidget(sub)

        # Cartes d'information
        cards_data = [
            (Icons.STRATEGY, "Architecture du Réseau Federe",
             "6 clients industriels : power, utilities, sap, pap, beneficiation, granulation.\n"
             "Modeles : IDSMLP (classification binaire) + IDSAutoencoder (anomalies zero-day).\n"
             "Agregation : FedAvg / FedProx via Flower (flwr). 19 features engineering."),

            (Icons.DATASET, "Datasets OT/ICS Supportes",
             "ICS-Flow, TON_IoT, ML-EdgeIIoT, X-IIoTID — partitionnes par client.\n"
             "Repertoire : datasets/partitions/<client_name>/train.csv et test.csv.\n"
             "Features : byte_rate, pkt_rate, flags TCP/ICS, ratios payload, etc."),

            (Icons.CLIENTS, "Clients FL (clients/client_app.py)",
             "IDSFlowerClient : implementation NumPyClient Flower.\n"
             "fit() : recoit parametres globaux → entrainemet local → retourne poids + n_samples.\n"
             "evaluate() : evaluation locale du modele global sur test set."),

            (Icons.GLOBE, "Serveur Central (central_server/)",
             "server.py et strategy.py : stubs vides a implementer.\n"
             "La strategie FedAvg est documentee mais pas encore codee.\n"
             "Formule : w_global = sum(n_k / N * w_k) pour chaque client k."),

            (Icons.BADGE, "Modeles (models/)",
             "IDSMLP : 19 -> 64 -> 32 -> 16 -> 1, BatchNorm + ReLU + Dropout(0.2).\n"
             "IDSAutoencoder : encodeur 19->64->32->16, decodeur inverse.\n"
             "HybridDetector : combine MLP + AE, 3 classes (normal, attaque, zero-day)."),

            (Icons.ROUNDS, "Prochaines Etapes",
             "1. Implementer central_server/server.py et strategy.py (Flower FedAvg).\n"
             "2. Brancher IDSFlowerClient dans LocalTrainerWorker (commentaires POINT D'INTEGRATION).\n"
             "3. Integrer metriques sklearn reels (Precision, Recall, AUC-ROC).\n"
             "4. Activer monitoring Prometheus → monitoring/metrics/"),
        ]

        for icon_svg, card_title, card_body in cards_data:
            card = self._make_info_card(icon_svg, card_title, card_body)
            main_layout.addWidget(card)

        main_layout.addStretch()
        scroll.setWidget(content)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

    def _make_info_card(self, icon_svg: str, title: str, body: str) -> QFrame:
        card = QFrame()
        card.setObjectName("sectionCard")
        card.setStyleSheet(
            "QFrame#sectionCard { background: #FFFFFF; border: 1px solid #E4E8F0; "
            "border-radius: 12px; }"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)

        # Titre avec icône
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        icon_lbl = QLabel()
        icon_lbl.setPixmap(svg_to_pixmap(icon_svg, 16, "#C9956A"))
        icon_lbl.setFixedSize(20, 20)
        title_row.addWidget(icon_lbl)
        t_lbl = QLabel(title)
        t_lbl.setStyleSheet(
            "color: #1A1D2E; font-size: 14px; font-weight: 700; "
            "background: transparent;"
        )
        title_row.addWidget(t_lbl)
        title_row.addStretch()
        layout.addLayout(title_row)

        # Séparateur
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #F0F2F7;")
        layout.addWidget(sep)

        # Corps
        body_lbl = QLabel(body)
        body_lbl.setStyleSheet(
            "color: #6B7280; font-size: 13px; line-height: 1.6; "
            "background: transparent;"
        )
        body_lbl.setWordWrap(True)
        layout.addWidget(body_lbl)

        return card


# ────────────────────────────────────────────────────────────────────────────
# Splitter fluide avec poignée visible
# ────────────────────────────────────────────────────────────────────────────
class GripHandle(QSplitterHandle):
    """
    Poignée de splitter personnalisée :
    - Fond visible au survol (rose-gold)
    - 5 points de grip centrés
    - Curseur SizeVer pour indiquer le redimensionnement
    """

    _COLOR_IDLE   = QColor("#E4E8F0")   # gris lavande — repos
    _COLOR_HOVER  = QColor("#C9956A")   # or rose — survol
    _COLOR_DOT_I  = QColor("#B0BCCF")   # points — repos
    _COLOR_DOT_H  = QColor("#FFFFFF")   # points — survol

    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self._hovered = False
        self.setFixedHeight(10)
        self.setCursor(Qt.CursorShape.SizeVerCursor)

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # ── Fond de la poignée ──────────────────────────────────────
        bg_color = self._COLOR_HOVER if self._hovered else self._COLOR_IDLE
        painter.fillRect(self.rect(), bg_color)

        # ── 5 points centrés ────────────────────────────────────────
        dot_color = self._COLOR_DOT_H if self._hovered else self._COLOR_DOT_I
        painter.setBrush(QBrush(dot_color))
        painter.setPen(QPen(Qt.PenStyle.NoPen))

        n_dots   = 5
        dot_r    = 2
        gap      = 7
        total_w  = n_dots * (dot_r * 2) + (n_dots - 1) * gap
        start_x  = (self.width()  - total_w) // 2
        center_y = self.height() // 2

        for i in range(n_dots):
            x = start_x + i * (dot_r * 2 + gap)
            painter.drawEllipse(x, center_y - dot_r, dot_r * 2, dot_r * 2)


class SmoothSplitter(QSplitter):
    """QSplitter avec GripHandle personnalisé."""

    def createHandle(self) -> QSplitterHandle:
        return GripHandle(self.orientation(), self)


# ────────────────────────────────────────────────────────────────────────────
# Page Dashboard : ChartsPanel + LogConsole
# ────────────────────────────────────────────────────────────────────────────
class DashboardPage(QWidget):
    """Encapsule ChartsPanel + LogConsole dans un SmoothSplitter vertical."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pageContainer")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = SmoothSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(True)   # autorise à masquer la console
        splitter.setStyleSheet("""
            QSplitter::handle { background: transparent; }
        """)

        self.charts = ChartsPanel()
        self.charts.setMinimumHeight(280)
        splitter.addWidget(self.charts)

        self.log_console = LogConsole()
        self.log_console.setMinimumHeight(60)   # permet de réduire beaucoup plus
        splitter.addWidget(self.log_console)

        # Proportion initiale : 65% graphiques / 35% console
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([580, 300])
        layout.addWidget(splitter)



# ────────────────────────────────────────────────────────────────────────────
# StatusBar personnalisée
# ────────────────────────────────────────────────────────────────────────────
class AppStatusBar(QFrame):
    """Barre de statut en bas de fenêtre (light mode)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setStyleSheet(
            "background: #FFFFFF; border-top: 1px solid #E4E8F0;"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(6)

        self._dot = QLabel("●")
        self._dot.setStyleSheet("color: #3D8B5F; font-size: 9px;")
        self._msg = QLabel("Pret — Aucun entrainement en cours")
        self._msg.setStyleSheet("color: #9CA3AF; font-size: 11px;")

        layout.addWidget(self._dot)
        layout.addWidget(self._msg)
        layout.addStretch()

        app_name = QLabel("FL-IDS-OT-ICS  v1.0")
        app_name.setStyleSheet("color: #CBD2E0; font-size: 11px;")
        layout.addWidget(app_name)

        self._time = QLabel()
        self._time.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        layout.addWidget(self._time)
        self._update_time()

        timer = QTimer(self)
        timer.timeout.connect(self._update_time)
        timer.start(1000)

    def _update_time(self):
        self._time.setText(datetime.now().strftime("  %H:%M:%S"))

    def set_running(self, mode: str):
        self._dot.setStyleSheet("color: #C9956A; font-size: 9px;")
        self._msg.setText(f"En cours : {mode}")

    def set_idle(self):
        self._dot.setStyleSheet("color: #3D8B5F; font-size: 9px;")
        self._msg.setText("Pret — Aucun entrainement en cours")

    def set_stopped(self):
        self._dot.setStyleSheet("color: #C0626A; font-size: 9px;")
        self._msg.setText("Interrompu manuellement")


# ────────────────────────────────────────────────────────────────────────────
# MainWindow
# ────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    """
    Fenêtre principale : NavSidebar | QStackedWidget (3 pages).
    Orchestre les workers QThread et connecte tous les signaux.
    """

    def __init__(self):
        super().__init__()
        self._fl_worker: FLRunnerWorker | None = None

        self._setup_window()
        self._build_ui()
        self._apply_stylesheet()

    # ────────────────────────────────────────────────────────────────────
    def _setup_window(self):
        self.setWindowTitle("FL-IDS-OT-ICS  —  Federated Learning Dashboard")
        self.setMinimumSize(QSize(1280, 800))
        self.resize(1440, 900)
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(
            max(0, (screen.width()  - 1440) // 2),
            max(0, (screen.height() - 900)  // 2),
        )

    def _apply_stylesheet(self):
        qss = _load_qss(_QSS_PATH)
        self.setStyleSheet(qss)

    # ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Widget central
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Ligne principale : sidebar | contenu ──────────────────────
        content_row = QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(0)

        # Sidebar de navigation
        self.nav = NavSidebar()
        self.nav.page_changed.connect(self._switch_page)
        content_row.addWidget(self.nav)

        # Zone de contenu principale
        self._stack = QStackedWidget()
        self._stack.setObjectName("pageContainer")

        # ── Page 0 : Dashboard ───────────────────────────────────────
        self.dashboard_page = DashboardPage()
        self._stack.addWidget(self.dashboard_page)

        # ── Page 1 : Panneau de Contrôle ─────────────────────────────
        self.control_panel = ControlPanel()
        self.control_panel.launch_fl_training.connect(self._start_fl_training)
        self.control_panel.emergency_stop.connect(self._stop_training)
        self._stack.addWidget(self.control_panel)

        # ── Page 2 : A propos ─────────────────────────────────────────
        self.about_page = AboutPage()
        self._stack.addWidget(self.about_page)

        content_row.addWidget(self._stack)
        root.addLayout(content_row)

        # ── Barre de statut ───────────────────────────────────────────
        self._status_bar = AppStatusBar(self)
        root.addWidget(self._status_bar)

        # Message de bienvenue dans la console du Dashboard
        self._log("FL-IDS-OT-ICS Dashboard demarre — framework operationnel.")
        self._log("Clients disponibles : power, utilities, sap, pap, beneficiation, granulation.")
        self._log("Accedez au Panneau de Controle via la navigation laterale pour lancer un entrainement.")

    # ── Navigation ────────────────────────────────────────────────────────
    @Slot(int)
    def _switch_page(self, index: int):
        self._stack.setCurrentIndex(index)

    # ── Lancement des entraînements ───────────────────────────────────────
    @Slot(dict)
    def _start_fl_training(self, params: dict):
        if self._is_busy():
            return
        self._log(f"Lancement de l'entrainement FL — {params}")
        self.dashboard_page.charts.reset_all()
        self.control_panel.update_progress(0, params["num_rounds"])

        self._fl_worker = FLRunnerWorker(params=params)
        
        # Connexions
        log_console = self.dashboard_page.log_console
        self._fl_worker.log_message.connect(log_console.append_log)
        self._fl_worker.metrics_update.connect(self.dashboard_page.charts.update_charts)
        self._fl_worker.metrics_update.connect(self.control_panel.update_metrics)
        self._fl_worker.progress_update.connect(self.control_panel.update_progress)
        self._fl_worker.finished.connect(self._on_fl_finished)
        self._fl_worker.error_occurred.connect(self._on_error)

        self.control_panel.set_training_state(True)
        self._status_bar.set_running(params.get("mode", "FL Training"))
        self._fl_worker.start()

    # ── Fin des entraînements ─────────────────────────────────────────────
    @Slot()
    def _on_fl_finished(self):
        self._cleanup()
        self.control_panel.set_training_state(False)
        self._status_bar.set_idle()
        self.control_panel.update_progress(
            self.control_panel.spin_rounds.value(),
            self.control_panel.spin_rounds.value(),
        )

    @Slot()
    def _stop_training(self):
        if hasattr(self, '_fl_worker') and self._fl_worker.isRunning():
            self._log("Arrêt d'urgence demandé...", "color: #C0626A; font-weight: bold;")
            self._fl_worker.stop()
            self._fl_worker.wait()
            self.control_panel.set_training_state(False)
            self._status_bar.set_idle()
        self.control_panel.update_progress(
            self.control_panel.spin_rounds.value(),
            self.control_panel.spin_rounds.value(),
        )

    @Slot(str)
    def _on_error(self, msg: str):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Erreur Worker", msg)
        self._cleanup()
        self.control_panel.set_training_state(False)
        self._status_bar.set_stopped()

    # ── Utilitaires ───────────────────────────────────────────────────────
    def _is_busy(self) -> bool:
        from PySide6.QtWidgets import QMessageBox
        if hasattr(self, '_fl_worker') and self._fl_worker is not None and self._fl_worker.isRunning():
            QMessageBox.warning(self, "En cours",
                "Un entrainement est deja en cours.\n"
                "Utilisez 'Arret d'urgence' pour l'interrompre.")
            return True
        return False

    def _cleanup(self):
        if hasattr(self, '_fl_worker') and self._fl_worker:
            if self._fl_worker.isRunning():
                self._fl_worker.stop()
                self._fl_worker.wait(3000)
            self._fl_worker.deleteLater()
        self._fl_worker = None

    def _log(self, msg: str, style: str = None):
        """Raccourci pour écrire dans la console du Dashboard."""
        self.dashboard_page.log_console.append_log(msg)

    def closeEvent(self, event):
        self._stop_training()
        if hasattr(self, '_fl_worker') and self._fl_worker:
            self._fl_worker.wait(2000)
        event.accept()
