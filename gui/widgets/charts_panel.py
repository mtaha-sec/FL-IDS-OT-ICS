"""
gui/widgets/charts_panel.py  (v3 — Light Mode + sans emojis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dashboard graphique temps réel — 4 graphiques sur la page Dashboard.
Thème light mode : fonds blancs/gris très clair, lignes fines.
Pas d'emojis dans les titres — icônes SVG dans les en-têtes de section.
"""

import math
import random

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont

from gui.icons import Icons, svg_to_pixmap

# ── pyqtgraph ────────────────────────────────────────────────────────────────
try:
    import pyqtgraph as pg
    from pyqtgraph import PlotWidget, mkPen, mkBrush

    pg.setConfigOptions(
        antialias=True,
        background="#FFFFFF",   # fond blanc en light mode
        foreground="#6B7280",   # texte gris
    )
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

# ── matplotlib ───────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ── Palette light mode ────────────────────────────────────────────────────────
PALETTE = {
    "global_loss":  "#C9956A",   # or rose  → loss globale
    "local_loss":   "#B0BCCF",   # gris-bleu atténué → loss locale
    "eval_loss":    "#C4964A",   # ambre → loss évaluation
    "accuracy":     "#3D8B5F",   # vert désaturé → accuracy
    "f1":           "#B07D2A",   # ambre foncé → F1-Score
    "teal":         "#4A8FA8",   # bleu canard → radar
    "grid":         "#F0F2F7",   # gris très clair → grille
    "text":         "#6B7280",   # gris → texte axes
    "danger":       "#C0626A",   # seuil 95%
}


def _make_plot(title: str) -> "pg.PlotWidget":
    """Crée un PlotWidget stylé light mode."""
    widget = pg.PlotWidget()
    widget.setBackground("#FAFBFD")
    widget.showGrid(x=True, y=True, alpha=0.5)

    widget.setTitle(title, color="#374151", size="11pt")

    for axis in ["bottom", "left"]:
        widget.getAxis(axis).setPen(pg.mkPen(color="#E4E8F0", width=1))
        widget.getAxis(axis).setTextPen(pg.mkPen(color="#9CA3AF"))
        widget.getAxis(axis).setStyle(tickFont=QFont("Segoe UI", 9))

    widget.setStyleSheet(
        "border: 1px solid #E4E8F0; border-radius: 8px;"
    )
    return widget


# ── Section header avec SVG ────────────────────────────────────────────────────
class ChartSectionHeader(QWidget):
    def __init__(self, icon_svg: str, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        icon_lbl = QLabel()
        icon_lbl.setPixmap(svg_to_pixmap(icon_svg, 14, "#C9956A"))
        icon_lbl.setFixedSize(18, 18)
        layout.addWidget(icon_lbl)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            "color: #374151; font-size: 12px; font-weight: 700; "
            "letter-spacing: 0.3px;"
        )
        layout.addWidget(title_lbl)
        layout.addStretch()


# ── Graphique 1 : Loss ────────────────────────────────────────────────────────
class LossChart(QWidget):
    """Evolution de la fonction de perte (Loss globale, locale, évaluation)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._global_losses: dict[int, float] = {}
        self._local_losses:  dict[int, float] = {}
        self._eval_losses:   dict[int, float] = {}
        self._client_min:    dict[int, float] = {}
        self._client_max:    dict[int, float] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(ChartSectionHeader(Icons.TRENDING_DOWN, "Evolution de la Loss"))

        if not PYQTGRAPH_AVAILABLE:
            layout.addWidget(QLabel("pyqtgraph non installe"))
            return

        self.plot = _make_plot("Loss federe")
        self.plot.setLabel("bottom", "Round", color="#9CA3AF")
        self.plot.setLabel("left",   "Loss",  color="#9CA3AF")

        # Bande incertitude clients
        self._fill_upper = pg.PlotDataItem(pen=None)
        self._fill_lower = pg.PlotDataItem(pen=None)
        self.plot.addItem(self._fill_upper)
        self.plot.addItem(self._fill_lower)
        self._fill = pg.FillBetweenItem(
            self._fill_upper, self._fill_lower,
            brush=pg.mkBrush(QColor(201, 149, 106, 20)),
        )
        self.plot.addItem(self._fill)

        self._curve_global = self.plot.plot(
            pen=mkPen(color=PALETTE["global_loss"], width=2.2),
            name="Loss Globale",
            symbol="o", symbolSize=4,
            symbolBrush=mkBrush(PALETTE["global_loss"]),
            symbolPen=mkPen(None),
        )
        self._curve_local = self.plot.plot(
            pen=mkPen(color=PALETTE["local_loss"], width=1.5, style=Qt.PenStyle.DashLine),
            name="Loss Locale (moy.)",
            symbol="s", symbolSize=3,
            symbolBrush=mkBrush(PALETTE["local_loss"]),
            symbolPen=mkPen(None),
        )
        self._curve_eval = self.plot.plot(
            pen=mkPen(color=PALETTE["eval_loss"], width=1.5, style=Qt.PenStyle.DotLine),
            name="Loss Evaluation",
            symbol="t", symbolSize=3,
            symbolBrush=mkBrush(PALETTE["eval_loss"]),
            symbolPen=mkPen(None),
        )

        self.plot.addLegend(
            offset=(10, 10),
            labelTextColor="#374151",
            brush=pg.mkBrush(QColor(255, 255, 255, 220)),
            pen=pg.mkPen(QColor(228, 232, 240)),
        )

        layout.addWidget(self.plot)

    def update(self, metrics: dict):
        if not PYQTGRAPH_AVAILABLE:
            return
        r = metrics.get("round")
        if r is None:
            return

        if "global_loss" in metrics:
            self._global_losses[r] = metrics["global_loss"]
        if "eval_loss" in metrics:
            self._eval_losses[r] = metrics["eval_loss"]
        if "client_losses" in metrics and metrics["client_losses"]:
            cl_list = metrics["client_losses"]
            self._local_losses[r] = sum(cl_list) / len(cl_list)
            self._client_min[r] = min(cl_list)
            self._client_max[r] = max(cl_list)

        if self._global_losses:
            r_g = sorted(self._global_losses.keys())
            self._curve_global.setData(r_g, [self._global_losses[x] for x in r_g])
        if self._eval_losses:
            r_e = sorted(self._eval_losses.keys())
            self._curve_eval.setData(r_e, [self._eval_losses[x] for x in r_e])
        if self._local_losses:
            r_l = sorted(self._local_losses.keys())
            self._curve_local.setData(r_l, [self._local_losses[x] for x in r_l])
            self._fill_upper.setData(r_l, [self._client_max[x] for x in r_l])
            self._fill_lower.setData(r_l, [self._client_min[x] for x in r_l])
            self._fill.setCurves(self._fill_upper, self._fill_lower)

    def reset(self):
        self._global_losses.clear()
        self._local_losses.clear()
        self._eval_losses.clear()
        self._client_min.clear()
        self._client_max.clear()
        if PYQTGRAPH_AVAILABLE:
            for c in [self._curve_global, self._curve_local, self._curve_eval,
                      self._fill_upper, self._fill_lower]:
                c.setData([], [])


# ── Graphique 2 : Accuracy & F1 ───────────────────────────────────────────────
class MetricsChart(QWidget):
    """Accuracy et F1-Score IDS par round."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._accuracies: dict[int, float] = {}
        self._f1_scores:  dict[int, float] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(ChartSectionHeader(Icons.ACCURACY, "Metriques IDS — Accuracy & F1"))

        if not PYQTGRAPH_AVAILABLE:
            layout.addWidget(QLabel("pyqtgraph non installe"))
            return

        self.plot = _make_plot("Accuracy & F1-Score")
        self.plot.setLabel("bottom", "Round", color="#9CA3AF")
        self.plot.setLabel("left",   "Score", color="#9CA3AF")
        self.plot.setYRange(0, 1.0, padding=0.05)

        # Seuil 95%
        self.plot.addItem(pg.InfiniteLine(
            pos=0.95, angle=0,
            pen=mkPen(color=PALETTE["danger"], width=1, style=Qt.PenStyle.DashLine),
            label="Seuil 95%",
            labelOpts={"color": PALETTE["danger"], "position": 0.02,
                       "anchors": [(0, 1), (0, 1)]},
        ))

        self._curve_acc = self.plot.plot(
            pen=mkPen(color=PALETTE["accuracy"], width=2.2),
            name="Accuracy",
            symbol="o", symbolSize=4,
            symbolBrush=mkBrush(PALETTE["accuracy"]),
            symbolPen=mkPen(None),
        )
        self._curve_f1 = self.plot.plot(
            pen=mkPen(color=PALETTE["f1"], width=2.2),
            name="F1-Score",
            symbol="d", symbolSize=4,
            symbolBrush=mkBrush(PALETTE["f1"]),
            symbolPen=mkPen(None),
        )

        self.plot.addLegend(
            offset=(10, 10),
            labelTextColor="#374151",
            brush=pg.mkBrush(QColor(255, 255, 255, 220)),
            pen=pg.mkPen(QColor(228, 232, 240)),
        )
        layout.addWidget(self.plot)

    def update(self, metrics: dict):
        if not PYQTGRAPH_AVAILABLE:
            return
        r = metrics.get("round")
        if r is None:
            return
            
        if "global_accuracy" in metrics:
            self._accuracies[r] = metrics["global_accuracy"]
        if "global_f1" in metrics:
            self._f1_scores[r] = metrics["global_f1"]
            
        if self._accuracies:
            r_a = sorted(self._accuracies.keys())
            self._curve_acc.setData(r_a, [self._accuracies[x] for x in r_a])
        if self._f1_scores:
            r_f = sorted(self._f1_scores.keys())
            self._curve_f1.setData(r_f, [self._f1_scores[x] for x in r_f])

    def reset(self):
        self._accuracies.clear()
        self._f1_scores.clear()
        if PYQTGRAPH_AVAILABLE:
            self._curve_acc.setData([], [])
            self._curve_f1.setData([], [])


# ── Graphique 3 : Barres par client ───────────────────────────────────────────
class ClientLossBar(QWidget):
    """Histogramme des losses par client (dernier round)."""

    CLIENT_NAMES = ["pow", "util", "sap", "pap", "ben", "gran"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bars: list = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(ChartSectionHeader(Icons.CLIENTS, "Loss par Client — Dernier Round"))

        if not PYQTGRAPH_AVAILABLE:
            layout.addWidget(QLabel("pyqtgraph non installe"))
            return

        self.plot = _make_plot("Loss clients")
        self.plot.setLabel("bottom", "Client FL", color="#9CA3AF")
        self.plot.setLabel("left",   "Loss",      color="#9CA3AF")
        self._x_axis = self.plot.getAxis("bottom")
        layout.addWidget(self.plot)

    def update(self, metrics: dict):
        if not PYQTGRAPH_AVAILABLE:
            return
        client_losses = metrics.get("client_losses", [])
        if not client_losses:
            return

        for bar in self._bars:
            self.plot.removeItem(bar)
        self._bars.clear()

        n     = len(client_losses)
        max_l = max(client_losses) or 1.0
        min_l = min(client_losses) or 0.0
        names = self.CLIENT_NAMES[:n] + [f"Client {i+1}" for i in range(n - len(self.CLIENT_NAMES))]

        for i, (loss, name) in enumerate(zip(client_losses, names)):
            ratio = (loss - min_l) / (max_l - min_l + 1e-8)
            # Vert sauge → brique désaturée (light mode)
            r = int(ratio * 160 + (1 - ratio) * 61)
            g = int(ratio * 82  + (1 - ratio) * 139)
            b = int(ratio * 82  + (1 - ratio) * 95)
            bar = pg.BarGraphItem(
                x=[i], height=[loss], width=0.55,
                brush=pg.mkBrush(QColor(r, g, b, 180)),
                pen=pg.mkPen(None),
            )
            self.plot.addItem(bar)
            self._bars.append(bar)

        ticks = [(i, name) for i, name in enumerate(names)]
        self._x_axis.setTicks([ticks])

    def reset(self):
        for bar in self._bars:
            self.plot.removeItem(bar)
        self._bars.clear()


# ── Graphique 4 : Radar matplotlib ───────────────────────────────────────────
class RadarChart(QWidget):
    """Spider chart des métriques IDS (Accuracy, F1, Precision, Recall, AUC, Det.Rate)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(ChartSectionHeader(Icons.BADGE, "Profil Metriques IDS (Radar)"))

        if not MATPLOTLIB_AVAILABLE:
            layout.addWidget(QLabel("matplotlib non installe"))
            return

        # Figure light mode
        self.fig = Figure(figsize=(4, 4), facecolor="#FAFBFD")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.ax = self.fig.add_subplot(111, polar=True, facecolor="#F4F6FA")
        self._style_ax()
        self._draw_empty()
        layout.addWidget(self.canvas)

    def _style_ax(self):
        self.ax.tick_params(colors="#9CA3AF", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#E4E8F0")

    def _draw_empty(self):
        self._draw_radar(
            ["Accuracy", "F1-Score", "Precision", "Recall", "AUC-ROC", "Det.Rate"],
            [0.0] * 6, "En attente"
        )

    def _draw_radar(self, categories: list, values: list, label: str = "FL Global"):
        if not MATPLOTLIB_AVAILABLE:
            return
        self.ax.clear()
        self.ax.set_facecolor("#F4F6FA")
        self._style_ax()

        n = len(categories)
        angles = [i * 2 * math.pi / n for i in range(n)] + [0]
        vals_c = values + [values[0]]

        for level in [0.25, 0.5, 0.75, 1.0]:
            self.ax.plot(angles, [level] * (n + 1), color="#E4E8F0", linewidth=0.7)

        # Teal désaturé en light mode
        self.ax.fill(angles[:-1], values, alpha=0.18, color="#4A8FA8")
        self.ax.plot(angles, vals_c, color="#4A8FA8", linewidth=2.0)
        self.ax.scatter(angles[:-1], values, color="#4A8FA8", s=30, zorder=5)

        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(categories, color="#6B7280", fontsize=8)
        self.ax.set_ylim(0, 1)
        self.ax.yaxis.set_visible(False)
        self.ax.set_title(f"Profil IDS  —  {label}", color="#C9956A",
                          fontsize=10, pad=18)
        self.canvas.draw()

    def update(self, metrics: dict):
        if not MATPLOTLIB_AVAILABLE:
            return
            
        # N'update le radar que si on a des métriques globales
        if "global_accuracy" not in metrics:
            return
            
        acc = metrics.get("global_accuracy", 0.0)
        f1  = metrics.get("global_f1", 0.0)
        pre = metrics.get("global_precision", acc * 0.95) # Fallback si absent
        rec = metrics.get("global_recall", f1 * 0.95)     # Fallback si absent
        
        # Simulé car non implémenté pour l'instant
        auc = acc * 0.98
        dr  = f1 * 0.99

        self._draw_radar(
            ["Accuracy", "F1-Score", "Precision", "Recall", "AUC-ROC", "Det.Rate"],
            [acc, f1, pre, rec, auc, dr],
            f"Round {metrics.get('round', '?')}"
        )

    def reset(self):
        if MATPLOTLIB_AVAILABLE:
            self.ax.clear()
            self._style_ax()
            self._draw_empty()


# ── Page Dashboard (4 graphiques) ─────────────────────────────────────────────
class ChartsPanel(QWidget):
    """
    Page Dashboard : 4 graphiques en grille 2×2 + en-tête de page.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pageContainer")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 16)
        layout.setSpacing(16)

        # ── En-tête de page ───────────────────────────────────────────
        header = QWidget()
        header.setStyleSheet("background: transparent;")
        h_layout = QVBoxLayout(header)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(4)

        title = QLabel("Dashboard")
        title.setObjectName("pageTitle")
        sub   = QLabel("Visualisation des métriques d'entraînement fédéré en temps réel")
        sub.setObjectName("pageSubtitle")

        h_layout.addWidget(title)
        h_layout.addWidget(sub)
        layout.addWidget(header)

        # ── Grille 2×2 de graphiques ──────────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(12)

        # Conteneurs graphiques blancs avec bordure
        def _chart_card(chart_widget: QWidget) -> QFrame:
            card = QFrame()
            card.setObjectName("sectionCard")
            card.setStyleSheet(
                "QFrame#sectionCard { background: #FFFFFF; "
                "border: 1px solid #E4E8F0; border-radius: 12px; }"
            )
            cl = QVBoxLayout(card)
            cl.setContentsMargins(14, 14, 14, 14)
            cl.addWidget(chart_widget)
            return card

        self.loss_chart    = LossChart()
        self.metrics_chart = MetricsChart()
        self.client_bar    = ClientLossBar()
        self.radar_chart   = RadarChart()

        grid.addWidget(_chart_card(self.loss_chart),    0, 0)
        grid.addWidget(_chart_card(self.metrics_chart), 0, 1)
        grid.addWidget(_chart_card(self.client_bar),    1, 0)
        grid.addWidget(_chart_card(self.radar_chart),   1, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        layout.addLayout(grid)

    def update_charts(self, metrics: dict):
        """Met à jour tous les graphiques depuis un metrics_update."""
        self.loss_chart.update(metrics)
        self.metrics_chart.update(metrics)
        self.client_bar.update(metrics)
        self.radar_chart.update(metrics)

    def reset_all(self):
        """Réinitialise tous les graphiques."""
        self.loss_chart.reset()
        self.metrics_chart.reset()
        self.client_bar.reset()
        self.radar_chart.reset()
