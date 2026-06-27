import sys
import numpy as np
from sympy import *
from sympy.abc import x
from sympy.calculus.util import continuous_domain
import warnings
warnings.filterwarnings("ignore")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QSplitter,
    QGroupBox, QGridLayout, QTabWidget, QScrollArea, QFrame,
    QComboBox, QDoubleSpinBox, QSpinBox, QStatusBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon, QSyntaxHighlighter, QTextCharFormat

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# ─── Stylesheet ────────────────────────────────────────────────────────────────

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0f1117;
    color: #e2e8f0;
    font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
    font-size: 13px;
}

QGroupBox {
    border: 1px solid #2d3748;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: 600;
    color: #a78bfa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #a78bfa;
}

QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #1a1f2e;
    border: 1px solid #374151;
    border-radius: 6px;
    padding: 6px 10px;
    color: #f1f5f9;
    selection-background-color: #6d28d9;
}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #7c3aed;
    background-color: #1e2438;
}

QPushButton {
    background-color: #6d28d9;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #7c3aed;
}
QPushButton:pressed {
    background-color: #5b21b6;
}
QPushButton#secondary {
    background-color: #1e2438;
    color: #a78bfa;
    border: 1px solid #374151;
}
QPushButton#secondary:hover {
    background-color: #252d42;
    border-color: #6d28d9;
}
QPushButton#danger {
    background-color: #1e2438;
    color: #f87171;
    border: 1px solid #374151;
}
QPushButton#danger:hover {
    background-color: #2a1a1a;
    border-color: #ef4444;
}

QTextEdit {
    background-color: #0d1117;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 10px;
    color: #e2e8f0;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.5;
}

QTabWidget::pane {
    border: 1px solid #2d3748;
    border-radius: 8px;
    background-color: #0f1117;
}
QTabBar::tab {
    background-color: #1a1f2e;
    color: #94a3b8;
    padding: 8px 20px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    font-weight: 500;
}
QTabBar::tab:selected {
    background-color: #6d28d9;
    color: #ffffff;
}
QTabBar::tab:hover:!selected {
    background-color: #252d42;
    color: #c4b5fd;
}

QSplitter::handle {
    background-color: #2d3748;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QStatusBar {
    background-color: #0d1117;
    color: #64748b;
    border-top: 1px solid #1e2438;
}

QLabel#title {
    font-size: 22px;
    font-weight: 700;
    color: #a78bfa;
}
QLabel#subtitle {
    font-size: 12px;
    color: #64748b;
}
QLabel#section {
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
}

QFrame#divider {
    background-color: #1e2438;
    max-height: 1px;
}
"""


# ─── Worker Thread ──────────────────────────────────────────────────────────────

class AnalysisWorker(QThread):
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, expr_str, x_min, x_max, limit_point, limit_dir):
        super().__init__()
        self.expr_str = expr_str
        self.x_min = x_min
        self.x_max = x_max
        self.limit_point = limit_point
        self.limit_dir = limit_dir

    def run(self):
        try:
            results = {}
            expr_str = self.expr_str.strip()

            # Parse expression
            local_dict = {"x": x, "e": E, "pi": pi, "inf": oo, "oo": oo}
            expr = sympify(expr_str, locals=local_dict)
            results["expr"] = expr

            # ── Derivative ──
            deriv = diff(expr, x)
            results["derivative"] = deriv

            # ── Second derivative ──
            deriv2 = diff(deriv, x)
            results["second_derivative"] = deriv2

            # ── Integral ──
            try:
                integ = integrate(expr, x)
                results["integral"] = integ
            except Exception:
                results["integral"] = "Could not compute"

            # ── Limit ──
            try:
                lp = sympify(self.limit_point, locals={"oo": oo, "inf": oo, "pi": pi})
                dir_map = {"Both (±)": None, "Right (+)": "+", "Left (−)": "-"}
                ldir = dir_map[self.limit_dir]

                if ldir is None:
                    lim_right = limit(expr, x, lp, "+")
                    lim_left  = limit(expr, x, lp, "-")
                    results["limit_right"] = lim_right
                    results["limit_left"]  = lim_left
                    results["limit"] = lim_right if lim_right == lim_left else "DNE (left ≠ right)"
                else:
                    lim_val = limit(expr, x, lp, ldir)
                    results["limit"] = lim_val
                    results["limit_right"] = lim_val if ldir == "+" else None
                    results["limit_left"]  = lim_val if ldir == "-" else None
            except Exception as e:
                results["limit"] = f"Error: {e}"

            # ── Critical points (peaks) ──
            try:
                crit_points = solve(deriv, x)
                peaks = []
                for cp in crit_points:
                    try:
                        cp_float = float(cp.evalf())
                        if self.x_min <= cp_float <= self.x_max:
                            y_val = float(expr.subs(x, cp).evalf())
                            d2_val = float(deriv2.subs(x, cp).evalf())
                            if d2_val < 0:
                                ptype = "Local Maximum"
                            elif d2_val > 0:
                                ptype = "Local Minimum"
                            else:
                                ptype = "Inflection / Saddle"
                            peaks.append((cp_float, y_val, ptype))
                    except Exception:
                        pass
                results["critical_points"] = peaks
            except Exception as e:
                results["critical_points"] = []

            # ── Inflection points ──
            try:
                infl_points = solve(deriv2, x)
                inflections = []
                for ip in infl_points:
                    try:
                        ip_f = float(ip.evalf())
                        if self.x_min <= ip_f <= self.x_max:
                            y_val = float(expr.subs(x, ip).evalf())
                            inflections.append((ip_f, y_val))
                    except Exception:
                        pass
                results["inflection_points"] = inflections
            except Exception:
                results["inflection_points"] = []

            # ── Domain analysis ──
            try:
                dom = continuous_domain(expr, x, S.Reals)
                results["domain"] = str(dom)
            except Exception:
                results["domain"] = "ℝ (assumed)"

            # ── Numerical plot data ──
            x_vals = np.linspace(self.x_min, self.x_max, 2000)
            f_lambda = lambdify(x, expr, modules=["numpy"])
            y_vals = f_lambda(x_vals)
            if not isinstance(y_vals, np.ndarray):
                y_vals = np.full_like(x_vals, float(y_vals))
            y_vals = np.where(np.isfinite(y_vals), y_vals, np.nan)
            results["x_vals"] = x_vals
            results["y_vals"] = y_vals

            self.result_ready.emit(results)

        except Exception as e:
            self.error_occurred.emit(str(e))


# ─── Plot Canvas ────────────────────────────────────────────────────────────────

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(facecolor="#0f1117")
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        self.fig.tight_layout(pad=1.5)

    def _style_axes(self):
        self.ax.set_facecolor("#0d1117")
        self.ax.tick_params(colors="#64748b", labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color("#2d3748")
        self.ax.xaxis.label.set_color("#94a3b8")
        self.ax.yaxis.label.set_color("#94a3b8")
        self.ax.grid(True, color="#1e2438", linewidth=0.7, linestyle="--")
        self.ax.set_title("", color="#a78bfa", fontsize=13, fontweight="bold")

    def plot(self, results, expr_str):
        self.ax.clear()
        self._style_axes()

        x_vals = results["x_vals"]
        y_vals = results["y_vals"]

        # Main curve
        self.ax.plot(x_vals, y_vals, color="#7c3aed", linewidth=2.2, label=f"f(x) = {expr_str}", zorder=3)

        # Derivative curve (lighter)
        try:
            deriv = results["derivative"]
            f_d = lambdify(x, deriv, modules=["numpy"])
            yd = f_d(x_vals)
            if isinstance(yd, np.ndarray):
                yd = np.where(np.isfinite(yd), yd, np.nan)
            else:
                yd = np.full_like(x_vals, float(yd))
            self.ax.plot(x_vals, yd, color="#06b6d4", linewidth=1.2,
                         linestyle="--", alpha=0.6, label="f'(x)", zorder=2)
        except Exception:
            pass

        # Critical points
        for cx, cy, ptype in results.get("critical_points", []):
            color = "#f87171" if "Max" in ptype else "#34d399" if "Min" in ptype else "#fbbf24"
            marker = "v" if "Max" in ptype else "^" if "Min" in ptype else "D"
            self.ax.scatter([cx], [cy], color=color, s=80, zorder=5, marker=marker)
            self.ax.annotate(f"  ({cx:.2f}, {cy:.2f})\n  {ptype}",
                             xy=(cx, cy), color=color, fontsize=8,
                             xytext=(8, 8), textcoords="offset points")

        # Inflection points
        for ix, iy in results.get("inflection_points", []):
            self.ax.scatter([ix], [iy], color="#fbbf24", s=50, zorder=4, marker="o",
                            edgecolors="#92400e", linewidths=1)

        # Zero line
        self.ax.axhline(0, color="#374151", linewidth=0.8, zorder=1)
        self.ax.axvline(0, color="#374151", linewidth=0.8, zorder=1)

        # Y-axis clip
        valid_y = y_vals[np.isfinite(y_vals)]
        if len(valid_y) > 0:
            q1, q99 = np.percentile(valid_y, 1), np.percentile(valid_y, 99)
            margin = max((q99 - q1) * 0.15, 0.5)
            self.ax.set_ylim(q1 - margin, q99 + margin)

        self.ax.set_xlabel("x", color="#94a3b8")
        self.ax.set_ylabel("f(x)", color="#94a3b8")
        self.ax.set_title(f"f(x) = {expr_str}", color="#a78bfa", fontsize=12, fontweight="bold", pad=10)
        legend = self.ax.legend(facecolor="#1a1f2e", edgecolor="#374151",
                                labelcolor="#e2e8f0", fontsize=9)

        self.fig.tight_layout(pad=1.5)
        self.draw()

    def clear_plot(self):
        self.ax.clear()
        self._style_axes()
        self.fig.tight_layout(pad=1.5)
        self.draw()


# ─── Main Window ────────────────────────────────────────────────────────────────

class MathAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Math Analyzer")
        self.setMinimumSize(1200, 750)
        self._worker = None
        self._build_ui()
        self.setStyleSheet(DARK_STYLE)
        self.status("Ready — enter a function and press Analyze")

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        root.addWidget(splitter)

        # Left panel
        left = QWidget()
        left.setMinimumWidth(320)
        left.setMaximumWidth(400)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(18, 18, 14, 18)
        left_layout.setSpacing(14)

        # Header
        title = QLabel("∫ Math Analyzer")
        title.setObjectName("title")
        left_layout.addWidget(title)

        sub = QLabel("Equations · Limits · Peaks · Calculus")
        sub.setObjectName("subtitle")
        left_layout.addWidget(sub)

        div = QFrame(); div.setObjectName("divider"); div.setFixedHeight(1)
        left_layout.addWidget(div)

        # Function input
        fn_group = QGroupBox("Function")
        fn_layout = QVBoxLayout(fn_group)

        hint = QLabel("EXPRESSION  —  use x as variable")
        hint.setObjectName("section")
        fn_layout.addWidget(hint)

        self.expr_input = QLineEdit()
        self.expr_input.setPlaceholderText("e.g.  x**3 - 3*x  or  sin(x)/x")
        self.expr_input.setFont(QFont("Consolas", 13))
        self.expr_input.returnPressed.connect(self.run_analysis)
        fn_layout.addWidget(self.expr_input)

        # Quick-insert buttons
        quick = QHBoxLayout()
        for sym in ["x²", "x³", "√x", "sin", "cos", "tan", "ln", "eˣ", "π"]:
            btn = QPushButton(sym)
            btn.setObjectName("secondary")
            btn.setFixedSize(QSize(38, 28))
            btn.setFont(QFont("Segoe UI", 10))
            sym_map = {
                "x²": "x**2", "x³": "x**3", "√x": "sqrt(x)",
                "sin": "sin(x)", "cos": "cos(x)", "tan": "tan(x)",
                "ln": "log(x)", "eˣ": "exp(x)", "π": "pi"
            }
            btn.clicked.connect(lambda _, s=sym_map[sym]: self._insert(s))
            quick.addWidget(btn)
        fn_layout.addLayout(quick)
        left_layout.addWidget(fn_group)

        # Range & Limit settings
        settings_group = QGroupBox("Range & Limit Settings")
        sg = QGridLayout(settings_group)
        sg.setColumnStretch(1, 1); sg.setColumnStretch(3, 1)

        sg.addWidget(QLabel("x min"), 0, 0)
        self.x_min = QDoubleSpinBox()
        self.x_min.setRange(-1e6, 1e6); self.x_min.setValue(-10); self.x_min.setSingleStep(1)
        sg.addWidget(self.x_min, 0, 1)

        sg.addWidget(QLabel("x max"), 0, 2)
        self.x_max = QDoubleSpinBox()
        self.x_max.setRange(-1e6, 1e6); self.x_max.setValue(10); self.x_max.setSingleStep(1)
        sg.addWidget(self.x_max, 0, 3)

        sg.addWidget(QLabel("Limit at x →"), 1, 0)
        self.limit_point = QLineEdit("0")
        self.limit_point.setFixedWidth(70)
        sg.addWidget(self.limit_point, 1, 1)

        sg.addWidget(QLabel("Direction"), 1, 2)
        self.limit_dir = QComboBox()
        self.limit_dir.addItems(["Both (±)", "Right (+)", "Left (−)"])
        sg.addWidget(self.limit_dir, 1, 3)

        left_layout.addWidget(settings_group)

        # Action buttons
        btn_row = QHBoxLayout()
        self.analyze_btn = QPushButton("⚡  Analyze")
        self.analyze_btn.setFixedHeight(38)
        self.analyze_btn.clicked.connect(self.run_analysis)
        btn_row.addWidget(self.analyze_btn)

        clear_btn = QPushButton("✕  Clear")
        clear_btn.setObjectName("danger")
        clear_btn.setFixedHeight(38)
        clear_btn.setFixedWidth(80)
        clear_btn.clicked.connect(self.clear_all)
        btn_row.addWidget(clear_btn)
        left_layout.addLayout(btn_row)

        # Example expressions
        ex_group = QGroupBox("Examples")
        ex_layout = QVBoxLayout(ex_group)
        examples = [
            ("Cubic", "x**3 - 3*x"),
            ("Rational", "1/(x**2 - 1)"),
            ("Sinc", "sin(x)/x"),
            ("Gaussian", "exp(-x**2)"),
            ("Oscillating", "x*sin(x)"),
            ("Polynomial", "x**4 - 4*x**2 + 2"),
        ]
        ex_grid = QGridLayout()
        for i, (name, expr) in enumerate(examples):
            btn = QPushButton(name)
            btn.setObjectName("secondary")
            btn.setToolTip(expr)
            btn.clicked.connect(lambda _, e=expr: self._load_example(e))
            ex_grid.addWidget(btn, i // 2, i % 2)
        ex_layout.addLayout(ex_grid)
        left_layout.addWidget(ex_group)

        left_layout.addStretch()
        splitter.addWidget(left)

        # Right panel (tabs)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 12, 18, 12)
        right_layout.setSpacing(8)

        tabs = QTabWidget()
        right_layout.addWidget(tabs)

        # Tab 1: Plot
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        plot_layout.setContentsMargins(0, 8, 0, 0)

        self.canvas = PlotCanvas()
        toolbar = NavigationToolbar(self.canvas, self)
        toolbar.setStyleSheet("background:#0f1117; color:#94a3b8; border:none;")
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.canvas)
        tabs.addTab(plot_tab, "📈  Graph")

        # Tab 2: Results
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        results_layout.setContentsMargins(0, 8, 0, 0)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Consolas", 12))
        results_layout.addWidget(self.results_text)
        tabs.addTab(results_tab, "🔢  Results")

        splitter.addWidget(right)
        splitter.setSizes([340, 860])

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def status(self, msg):
        self.status_bar.showMessage(msg)

    def _insert(self, text):
        cur = self.expr_input.text()
        self.expr_input.setText(cur + text)
        self.expr_input.setFocus()

    def _load_example(self, expr):
        self.expr_input.setText(expr)
        self.run_analysis()

    # ── Analysis ─────────────────────────────────────────────────────────────

    def run_analysis(self):
        expr_str = self.expr_input.text().strip()
        if not expr_str:
            self.status("⚠  Please enter a function first")
            return

        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("⏳  Computing…")
        self.status(f"Analyzing  f(x) = {expr_str}  …")

        self._worker = AnalysisWorker(
            expr_str,
            self.x_min.value(),
            self.x_max.value(),
            self.limit_point.text().strip() or "0",
            self.limit_dir.currentText()
        )
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def _on_result(self, results):
        expr_str = self.expr_input.text().strip()

        # Plot
        try:
            self.canvas.plot(results, expr_str)
        except Exception as e:
            self.status(f"Plot error: {e}")

        # Results text
        lp = self.limit_point.text().strip() or "0"
        ldir = self.limit_dir.currentText()
        lines = []
        lines.append(f"{'═'*54}")
        lines.append(f"  f(x) = {results['expr']}")
        lines.append(f"{'═'*54}")
        lines.append("")
        lines.append("  DERIVATIVE")
        lines.append(f"  f′(x) = {results['derivative']}")
        lines.append(f"  f″(x) = {results['second_derivative']}")
        lines.append("")
        lines.append("  INTEGRAL")
        lines.append(f"  ∫f(x) dx = {results['integral']}  + C")
        lines.append("")
        lines.append(f"  LIMIT  (x → {lp},  {ldir})")

        if isinstance(results["limit"], str) and results["limit"].startswith("DNE"):
            lines.append(f"  lim f(x)  =  Does Not Exist")
            if results.get("limit_right") is not None:
                lines.append(f"    Right:  {results['limit_right']}")
            if results.get("limit_left") is not None:
                lines.append(f"    Left:   {results['limit_left']}")
        else:
            lines.append(f"  lim f(x)  =  {results['limit']}")
            if ldir == "Both (±)":
                if results.get("limit_right") is not None:
                    lines.append(f"    Right (+): {results['limit_right']}")
                if results.get("limit_left") is not None:
                    lines.append(f"    Left  (−): {results['limit_left']}")

        lines.append("")
        lines.append("  CRITICAL POINTS  (peaks in range)")
        cps = results.get("critical_points", [])
        if cps:
            for cx, cy, ptype in cps:
                lines.append(f"    x = {cx:.6f}   f(x) = {cy:.6f}   →  {ptype}")
        else:
            lines.append("    No critical points found in range")

        lines.append("")
        lines.append("  INFLECTION POINTS")
        ips = results.get("inflection_points", [])
        if ips:
            for ix, iy in ips:
                lines.append(f"    x = {ix:.6f}   f(x) = {iy:.6f}")
        else:
            lines.append("    None in range")

        lines.append("")
        lines.append("  DOMAIN")
        lines.append(f"    {results.get('domain', 'ℝ')}")
        lines.append("")
        lines.append(f"{'─'*54}")

        self.results_text.setPlainText("\n".join(lines))

        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("⚡  Analyze")
        self.status(f"✓  Analysis complete for  f(x) = {expr_str}")

    def _on_error(self, msg):
        self.results_text.setPlainText(
            f"{'═'*50}\n  ERROR\n{'═'*50}\n\n  {msg}\n\n"
            "  Tips:\n"
            "  • Use ** for powers:  x**2  not  x^2\n"
            "  • Use * for multiply: 2*x   not  2x\n"
            "  • Available: sin, cos, tan, log, exp, sqrt, abs\n"
            "  • Constants: pi, e, oo (infinity)\n"
        )
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("⚡  Analyze")
        self.status(f"✗  Error parsing expression")

    def clear_all(self):
        self.expr_input.clear()
        self.results_text.clear()
        self.canvas.clear_plot()
        self.status("Cleared")


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Math Analyzer")
    window = MathAnalyzer()
    window.show()
    sys.exit(app.exec())
