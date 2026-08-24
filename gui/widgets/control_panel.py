"""
gui/widgets/control_panel.py  (v3 — Light Mode + SVG + Page)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Page "Panneau de Contrôle" : métriques temps réel, configuration
des hyperparamètres, boutons d'actions. Sans emojis — icônes SVG.

Signaux émis vers MainWindow :
    launch_fl_training (dict)
    emergency_stop        ()
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QComboBox, QFrame, QSizePolicy, QProgressBar, QScrollArea,
    QSplitter, QSplitterHandle, QCheckBox,
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont, QColor, QPainter, QBrush, QPen

from gui.icons import Icons, svg_to_pixmap, svg_icon


# ── Couleurs pour les icônes dans la zone claire ─────────────────────────────
IC_ACCENT   = "#C9956A"
IC_DEFAULT  = "#6B7280"
IC_SUCCESS  = "#3D8B5F"
IC_DANGER   = "#C0626A"


# ────────────────────────────────────────────────────────────────────────────
# Helper : en-tête de section avec icône SVG
# ────────────────────────────────────────────────────────────────────────────
class SectionHeader(QWidget):
    """
    Ligne titre de section : [icône SVG] [texte] ─────────────
    Remplace les GroupBox avec emojis dans le titre.
    """

    def __init__(self, icon_svg: str, title: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Icône
        icon_lbl = QLabel()
        icon_lbl.setPixmap(svg_to_pixmap(icon_svg, 16, IC_ACCENT))
        icon_lbl.setFixedSize(20, 20)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_lbl)

        # Texte du titre
        title_lbl = QLabel(title.upper())
        title_lbl.setObjectName("sectionTitle")
        title_lbl.setStyleSheet(
            "color: #374151; font-size: 11px; font-weight: 700; "
            "letter-spacing: 1px;"
        )
        layout.addWidget(title_lbl)

        # Ligne séparatrice
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #E4E8F0;")
        line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(line)


# ────────────────────────────────────────────────────────────────────────────
# Carte de métrique temps réel
# ────────────────────────────────────────────────────────────────────────────
class MetricCard(QFrame):
    """
    Carte blanche avec icône SVG, valeur mise en évidence et label.
    """

    def __init__(self, icon_svg: str, label: str, unit: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.unit = unit

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 12)
        layout.setSpacing(4)

        # Icône
        icon_row = QHBoxLayout()
        icon_row.setContentsMargins(0, 0, 0, 0)
        icon_lbl = QLabel()
        icon_lbl.setPixmap(svg_to_pixmap(icon_svg, 18, IC_ACCENT))
        icon_row.addWidget(icon_lbl)
        icon_row.addStretch()
        layout.addLayout(icon_row)

        layout.addSpacing(6)

        # Valeur
        self.value_label = QLabel("—")
        self.value_label.setObjectName("metricValue")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.value_label.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)

        # Label
        self.name_label = QLabel(label)
        self.name_label.setObjectName("metricLabel")
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(self.value_label)
        layout.addWidget(self.name_label)

    def update_value(self, value: float, precision: int = 4):
        """Met à jour la valeur affichée."""
        if isinstance(value, float):
            text = f"{value:.{precision}f}{self.unit}"
        else:
            text = f"{value}{self.unit}"
        self.value_label.setText(text)


# ────────────────────────────────────────────────────────────────────────────
# Widget de champ de formulaire avec icône SVG
# ────────────────────────────────────────────────────────────────────────────
class FormRow(QWidget):
    """
    Ligne de formulaire : [icône SVG] [label] + [widget input]
    Disposition horizontale ou verticale selon le contexte.
    """

    def __init__(self, icon_svg: str, label: str, input_widget: QWidget, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Label row
        label_row = QHBoxLayout()
        label_row.setContentsMargins(0, 0, 0, 0)
        label_row.setSpacing(6)

        icon_lbl = QLabel()
        icon_lbl.setPixmap(svg_to_pixmap(icon_svg, 14, IC_DEFAULT))
        icon_lbl.setFixedSize(16, 16)
        label_row.addWidget(icon_lbl)

        lbl = QLabel(label)
        lbl.setObjectName("fieldLabel")
        lbl.setStyleSheet("color: #374151; font-size: 13px; font-weight: 500;")
        label_row.addWidget(lbl)
        label_row.addStretch()

        layout.addLayout(label_row)
        layout.addWidget(input_widget)


# ────────────────────────────────────────────────────────────────────────────
# Splitter fluide local (identique à app.py mais autonome)
# ────────────────────────────────────────────────────────────────────────────
class _GripHandle(QSplitterHandle):
    _C_IDLE  = QColor("#E4E8F0")
    _C_HOV   = QColor("#C9956A")
    _C_DOT_I = QColor("#B0BCCF")
    _C_DOT_H = QColor("#FFFFFF")

    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self._h = False
        self.setFixedHeight(10)
        self.setCursor(Qt.CursorShape.SizeVerCursor)

    def enterEvent(self, e): self._h = True;  self.update(); super().enterEvent(e)
    def leaveEvent(self, e): self._h = False; self.update(); super().leaveEvent(e)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), self._C_HOV if self._h else self._C_IDLE)
        p.setBrush(QBrush(self._C_DOT_H if self._h else self._C_DOT_I))
        p.setPen(QPen(Qt.PenStyle.NoPen))
        r, g, n = 2, 7, 5
        tw = n * (r * 2) + (n - 1) * g
        sx = (self.width() - tw) // 2
        cy = self.height() // 2
        for i in range(n):
            p.drawEllipse(sx + i * (r * 2 + g), cy - r, r * 2, r * 2)


class _SmoothSplitter(QSplitter):
    def createHandle(self):
        return _GripHandle(self.orientation(), self)


class ControlPanel(QWidget):
    """
    Page complète de contrôle : métriques + configuration + actions.

    Signaux :
        launch_local_training (dict)
        launch_global_agg     (dict)
        emergency_stop        ()
    """

    launch_fl_training = Signal(dict)
    emergency_stop        = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pageContainer")
        self._build_ui()

    def _build_ui(self):
        # ── ScrollArea unique
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: #F4F6FA; border: none; }")

        content = QWidget()
        content.setStyleSheet("background: #F4F6FA;")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        layout.addWidget(self._build_page_header())
        layout.addWidget(SectionHeader(Icons.LOSS_CHART, "Métriques Temps Réel"))
        layout.addWidget(self._build_metric_cards())
        layout.addWidget(SectionHeader(Icons.SETTINGS, "Configuration des Hyperparamètres"))
        layout.addWidget(self._build_config_card())
        layout.addWidget(SectionHeader(Icons.PLAY, "Progression"))
        layout.addWidget(self._build_progress_card())
        layout.addStretch()

        scroll.setWidget(content)

        # Layout racine
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(scroll)


    # ── Sous-constructeurs ────────────────────────────────────────────────
    def _build_page_header(self) -> QWidget:
        widget = QWidget()
        widget.setStyleSheet("background: transparent;")
        main_layout = QHBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)
        title = QLabel("Panneau de Contrôle")
        title.setObjectName("pageTitle")
        sub = QLabel("Configurez et lancez vos entraînements fédérés FL-IDS")
        sub.setObjectName("pageSubtitle")
        text_layout.addWidget(title)
        text_layout.addWidget(sub)
        
        main_layout.addLayout(text_layout)
        main_layout.addStretch()
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        
        self.btn_launch_fl = QPushButton("  Train")
        self.btn_launch_fl.setObjectName("btnLaunchFL")
        self.btn_launch_fl.setMinimumHeight(38)
        self.btn_launch_fl.setIcon(svg_icon(Icons.PLAY, 16, "#FFFFFF"))
        self.btn_launch_fl.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_launch_fl.setStyleSheet("QPushButton#btnLaunchFL { padding: 0 16px; font-weight: bold; background-color: #C9956A; color: white; border-radius: 8px; } QPushButton#btnLaunchFL:hover { background-color: #B5845C; } QPushButton#btnLaunchFL:disabled { background-color: #E4E8F0; color: #9CA3AF; }")
        self.btn_launch_fl.setToolTip("Lancer l'Apprentissage Fédéré")
        self.btn_launch_fl.clicked.connect(self._on_launch_fl)
        
        self.btn_stop = QPushButton("  Stop")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setMinimumHeight(38)
        self.btn_stop.setIcon(svg_icon(Icons.STOP, 14, IC_DANGER))
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("QPushButton#btnStop { padding: 0 16px; font-weight: bold; color: #DC2626; background-color: #FEE2E2; border-radius: 8px; } QPushButton#btnStop:hover { background-color: #FCA5A5; } QPushButton#btnStop:disabled { background-color: #F3F4F6; color: #9CA3AF; }")
        self.btn_stop.setToolTip("Arrêt d'urgence")
        self.btn_stop.clicked.connect(self._on_emergency_stop)
        
        btn_layout.addWidget(self.btn_launch_fl)
        btn_layout.addWidget(self.btn_stop)
        
        main_layout.addLayout(btn_layout)
        return widget

    def _build_metric_cards(self) -> QWidget:
        widget = QWidget()
        grid = QGridLayout(widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)

        self.card_round    = MetricCard(Icons.EPOCHS,          "Round")
        self.card_loss     = MetricCard(Icons.LOSS_CHART,      "Loss Globale")
        self.card_accuracy = MetricCard(Icons.ACCURACY,        "Accuracy",       unit="%")
        self.card_f1       = MetricCard(Icons.F1_SCORE,        "F1-Score",       unit="%")

        for card in [self.card_round, self.card_loss, self.card_accuracy, self.card_f1]:
            card.setMinimumHeight(120)

        grid.addWidget(self.card_round, 0, 0)
        grid.addWidget(self.card_loss, 0, 1)
        grid.addWidget(self.card_accuracy, 1, 0)
        grid.addWidget(self.card_f1, 1, 1)

        return widget

    def _build_config_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("sectionCard")
        card.setStyleSheet(
            "QFrame#sectionCard { background: #FFFFFF; border: 1px solid #E4E8F0; "
            "border-radius: 12px; }"
        )
        grid = QGridLayout(card)
        grid.setContentsMargins(20, 20, 20, 20)
        grid.setSpacing(16)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # ── Ligne 0 : Rounds + FedProx µ ───────────────────────────────
        self.spin_rounds = QSpinBox()
        self.spin_rounds.setRange(1, 200)
        self.spin_rounds.setValue(10)
        self.spin_rounds.setToolTip("Nombre de rounds de communication fédérés (T)")
        grid.addWidget(FormRow(Icons.ROUNDS, "Rounds de communication", self.spin_rounds), 0, 0)

        self.spin_mu = QDoubleSpinBox()
        self.spin_mu.setRange(0.0, 5.0)
        self.spin_mu.setValue(0.1)
        self.spin_mu.setSingleStep(0.1)
        self.spin_mu.setDecimals(3)
        self.spin_mu.setToolTip("Terme proximal µ pour FedProx")
        grid.addWidget(FormRow(Icons.STRATEGY, "Paramètre FedProx (µ)", self.spin_mu), 0, 1)

        # ── Ligne 1 : Epochs + LR ────────────────────────────────────
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 100)
        self.spin_epochs.setValue(3)
        self.spin_epochs.setToolTip("Epochs d'entraînement local par client (E)")
        grid.addWidget(FormRow(Icons.EPOCHS, "Epochs locaux", self.spin_epochs), 1, 0)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 1.0)
        self.spin_lr.setValue(0.01)
        self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setToolTip("Taux d'apprentissage (η)")
        grid.addWidget(FormRow(Icons.LEARNING_RATE, "Learning Rate (η)", self.spin_lr), 1, 1)

        # ── Ligne 2 : Sélection des clients ───────────────────────────
        clients_lbl = QLabel("Sites industriels (Clients) :")
        clients_lbl.setStyleSheet("font-weight: 500; color: #1A1D2E;")
        grid.addWidget(clients_lbl, 2, 0, 1, 2)

        self.client_checkboxes = {}
        client_names = ["beneficiation", "sap", "pap", "power", "utilities", "granulation"]
        
        clients_widget = QWidget()
        clients_layout = QGridLayout(clients_widget)
        clients_layout.setContentsMargins(0, 0, 0, 0)
        
        for i, name in enumerate(client_names):
            cb = QCheckBox(name.capitalize())
            cb.setChecked(True)  # cochés par défaut
            self.client_checkboxes[name] = cb
            clients_layout.addWidget(cb, i // 3, i % 3)
            
        grid.addWidget(clients_widget, 3, 0, 1, 2)

        return card



    def _build_progress_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("sectionCard")
        card.setStyleSheet(
            "QFrame#sectionCard { background: #FFFFFF; border: 1px solid #E4E8F0; "
            "border-radius: 12px; }"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("En attente d'un lancement...")
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Inactif — aucun entraînement en cours")
        self.status_label.setObjectName("labelStatus")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        return card

    # ── Slots internes ────────────────────────────────────────────────────
    def _on_launch_fl(self):
        self.launch_fl_training.emit(self._collect_params())

    def _on_emergency_stop(self):
        self.emergency_stop.emit()

    def _collect_params(self) -> dict:
        selected_clients = [name for name, cb in self.client_checkboxes.items() if cb.isChecked()]
        return {
            "num_rounds":           self.spin_rounds.value(),
            "selected_clients":     selected_clients,
            "local_epochs":         self.spin_epochs.value(),
            "learning_rate":        self.spin_lr.value(),
            "mu":                   self.spin_mu.value(),
        }

    # ── API publique ──────────────────────────────────────────────────────
    def set_training_state(self, is_running: bool):
        """Désactive/réactive les contrôles selon l'état d'entraînement."""
        self.btn_launch_fl.setEnabled(not is_running)
        self.btn_stop.setEnabled(is_running)
        self.spin_rounds.setEnabled(not is_running)
        self.spin_epochs.setEnabled(not is_running)
        self.spin_lr.setEnabled(not is_running)
        self.spin_mu.setEnabled(not is_running)
        
        for cb in self.client_checkboxes.values():
            cb.setEnabled(not is_running)

        if is_running:
            self.status_label.setObjectName("labelStatusRunning")
            self.status_label.setText("En cours d'exécution...")
        else:
            self.status_label.setObjectName("labelStatus")
            self.status_label.setText("Inactif — aucun entraînement en cours")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def update_metrics(self, metrics: dict):
        """Met à jour les cartes de métriques temps réel."""
        if "round" in metrics:
            self.card_round.update_value(metrics["round"], precision=0)
        if "global_loss" in metrics:
            self.card_loss.update_value(metrics["global_loss"])
        if "global_accuracy" in metrics:
            self.card_accuracy.update_value(metrics["global_accuracy"] * 100, precision=2)
        if "global_f1" in metrics:
            self.card_f1.update_value(metrics["global_f1"] * 100, precision=2)

    def update_progress(self, current: int, total: int):
        """Met à jour la barre de progression."""
        percent = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"Round {current}/{total}  —  {percent}%")


# Import QSize ici pour les icônes des boutons
from PySide6.QtCore import QSize
