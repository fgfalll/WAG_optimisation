"""
Subsurface Studio Icon Generator.
=================================

Provides crisp, high-DPI vector-drawn icons for the Hierarchical Model Tree and HUD toolbars.
Generates native QIcon instances for all subsurface domains:
- Reservoir Geometry / Grid
- Geological Stratigraphy & Layers
- Petrophysics & Corey Rel-Perm
- Fluids & PVT Miscibility
- Wellbore Networks (Producers & Injectors)
- Geomechanics & Faults
- EPA Class VI Containment & Visual Audit
"""

from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QBrush, QPolygonF, QFont
from PyQt6.QtCore import Qt, QPointF, QRectF


def create_subsurface_icon(icon_type: str, size: int = 16) -> QIcon:
    """Generates a crisp, professional 16x16 or 24x24 QIcon for tree items and buttons."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    s = float(size)

    if icon_type in ("reservoir", "grid"):
        # 3D Isometric Cube Grid
        painter.setPen(QPen(QColor("#0284c7"), 1.2))
        painter.setBrush(QBrush(QColor("#e0f2fe")))
        # Top face
        top = QPolygonF([QPointF(s*0.5, s*0.1), QPointF(s*0.9, s*0.3), QPointF(s*0.5, s*0.5), QPointF(s*0.1, s*0.3)])
        painter.drawPolygon(top)
        # Left face
        painter.setBrush(QBrush(QColor("#bae6fd")))
        left = QPolygonF([QPointF(s*0.1, s*0.3), QPointF(s*0.5, s*0.5), QPointF(s*0.5, s*0.9), QPointF(s*0.1, s*0.7)])
        painter.drawPolygon(left)
        # Right face
        painter.setBrush(QBrush(QColor("#7dd3fc")))
        right = QPolygonF([QPointF(s*0.5, s*0.5), QPointF(s*0.9, s*0.3), QPointF(s*0.9, s*0.7), QPointF(s*0.5, s*0.9)])
        painter.drawPolygon(right)

    elif icon_type == "volumetrics":
        # Reservoir Tank / Volume Cylinder
        painter.setPen(QPen(QColor("#2563eb"), 1.2))
        painter.setBrush(QBrush(QColor("#dbeafe")))
        painter.drawRoundedRect(QRectF(s*0.15, s*0.2, s*0.7, s*0.65), 3, 3)
        painter.setPen(QPen(QColor("#1d4ed8"), 1.0))
        painter.drawLine(QPointF(s*0.2, s*0.45), QPointF(s*0.8, s*0.45))
        painter.drawLine(QPointF(s*0.2, s*0.65), QPointF(s*0.8, s*0.65))

    elif icon_type in ("stratigraphy", "layers"):
        # 3 Geological Layers
        layers = [
            (s*0.15, s*0.22, "#d97706", "#fef3c7"),  # Top Sand
            (s*0.42, s*0.22, "#475569", "#e2e8f0"),  # Mid Shale
            (s*0.69, s*0.22, "#0d9488", "#ccfbf1"),  # Bot Carbonate
        ]
        for y, h, pen_col, brush_col in layers:
            painter.setPen(QPen(QColor(pen_col), 1.0))
            painter.setBrush(QBrush(QColor(brush_col)))
            painter.drawRoundedRect(QRectF(s*0.1, y, s*0.8, h), 2, 2)

    elif icon_type in ("rock", "petrophysics"):
        # Rock Core Cylinder with porous texture
        painter.setPen(QPen(QColor("#ea580c"), 1.2))
        painter.setBrush(QBrush(QColor("#ffedd5")))
        painter.drawRoundedRect(QRectF(s*0.2, s*0.15, s*0.6, s*0.7), 4, 4)
        painter.setPen(QPen(QColor("#c2410c"), 1.5))
        painter.drawPoint(QPointF(s*0.35, s*0.35))
        painter.drawPoint(QPointF(s*0.65, s*0.45))
        painter.drawPoint(QPointF(s*0.45, s*0.65))

    elif icon_type == "relperm":
        # Two crossing Corey curves
        painter.setPen(QPen(QColor("#cbd5e1"), 1.0))
        painter.drawRect(QRectF(s*0.1, s*0.1, s*0.8, s*0.8))
        # Water curve (blue)
        painter.setPen(QPen(QColor("#0d6efd"), 1.5))
        painter.drawLine(QPointF(s*0.15, s*0.85), QPointF(s*0.85, s*0.35))
        # Oil curve (green)
        painter.setPen(QPen(QColor("#16a34a"), 1.5))
        painter.drawLine(QPointF(s*0.15, s*0.25), QPointF(s*0.85, s*0.85))

    elif icon_type == "geostat":
        # Variogram curve with scatter points
        painter.setPen(QPen(QColor("#9333ea"), 1.5))
        painter.drawArc(QRectF(s*0.1, s*0.1, s*1.2, s*1.4), 90 * 16, 90 * 16)
        painter.setPen(QPen(QColor("#e11d48"), 2.0))
        painter.drawPoint(QPointF(s*0.25, s*0.75))
        painter.drawPoint(QPointF(s*0.5, s*0.45))
        painter.drawPoint(QPointF(s*0.75, s*0.3))

    elif icon_type in ("pvt", "fluids"):
        # Liquid Droplet
        painter.setPen(QPen(QColor("#4f46e5"), 1.2))
        painter.setBrush(QBrush(QColor("#e0e7ff")))
        drop = QPolygonF([
            QPointF(s*0.5, s*0.1),
            QPointF(s*0.8, s*0.55),
            QPointF(s*0.7, s*0.85),
            QPointF(s*0.3, s*0.85),
            QPointF(s*0.2, s*0.55)
        ])
        painter.drawPolygon(drop)

    elif icon_type == "mmp":
        # Pressure Gauge Barometer
        painter.setPen(QPen(QColor("#059669"), 1.2))
        painter.setBrush(QBrush(QColor("#d1fae5")))
        painter.drawEllipse(QRectF(s*0.15, s*0.15, s*0.7, s*0.7))
        painter.setPen(QPen(QColor("#dc2626"), 1.5))
        painter.drawLine(QPointF(s*0.5, s*0.5), QPointF(s*0.7, s*0.3))

    elif icon_type in ("wells", "derrick"):
        # Oil Derrick Tower
        painter.setPen(QPen(QColor("#0f172a"), 1.3))
        # Derrick legs
        painter.drawLine(QPointF(s*0.2, s*0.9), QPointF(s*0.4, s*0.15))
        painter.drawLine(QPointF(s*0.8, s*0.9), QPointF(s*0.6, s*0.15))
        painter.drawLine(QPointF(s*0.4, s*0.15), QPointF(s*0.6, s*0.15))
        # Cross braces
        painter.drawLine(QPointF(s*0.25, s*0.65), QPointF(s*0.75, s*0.65))
        painter.drawLine(QPointF(s*0.33, s*0.4), QPointF(s*0.67, s*0.4))
        painter.drawLine(QPointF(s*0.25, s*0.65), QPointF(s*0.67, s*0.4))
        painter.drawLine(QPointF(s*0.75, s*0.65), QPointF(s*0.33, s*0.4))

    elif icon_type == "well_prod":
        # Green circle with 'P'
        painter.setPen(QPen(QColor("#15803d"), 1.2))
        painter.setBrush(QBrush(QColor("#86efac")))
        painter.drawEllipse(QRectF(s*0.15, s*0.15, s*0.7, s*0.7))
        painter.setPen(QPen(QColor("#14532d"), 1.0))
        font = QFont("Segoe UI", int(s*0.45), QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(0, 0, s, s), Qt.AlignmentFlag.AlignCenter, "P")

    elif icon_type == "well_inj":
        # Red circle with 'I'
        painter.setPen(QPen(QColor("#b91c1c"), 1.2))
        painter.setBrush(QBrush(QColor("#fca5a5")))
        painter.drawEllipse(QRectF(s*0.15, s*0.15, s*0.7, s*0.7))
        painter.setPen(QPen(QColor("#7f1d1d"), 1.0))
        font = QFont("Segoe UI", int(s*0.45), QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(QRectF(0, 0, s, s), Qt.AlignmentFlag.AlignCenter, "I")

    elif icon_type in ("faults", "geomechanics"):
        # Tectonic Fault Fracture Line
        painter.setPen(QPen(QColor("#be123c"), 1.5))
        painter.drawLine(QPointF(s*0.25, s*0.9), QPointF(s*0.75, s*0.1))
        # Arrows showing displacement
        painter.setPen(QPen(QColor("#e11d48"), 1.2))
        painter.drawLine(QPointF(s*0.35, s*0.35), QPointF(s*0.35, s*0.55))
        painter.drawLine(QPointF(s*0.65, s*0.65), QPointF(s*0.65, s*0.45))

    elif icon_type == "caprock":
        # Impermeable Caprock Seal Barrier
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor("#64748b")))
        # Upper confining seal block
        painter.drawRoundedRect(QRectF(s*0.1, s*0.15, s*0.8, s*0.35), 2, 2)
        # Seal hash lines
        painter.setPen(QPen(QColor("#f8fafc"), 1.0))
        painter.drawLine(QPointF(s*0.2, s*0.22), QPointF(s*0.4, s*0.42))
        painter.drawLine(QPointF(s*0.4, s*0.22), QPointF(s*0.6, s*0.42))
        painter.drawLine(QPointF(s*0.6, s*0.22), QPointF(s*0.8, s*0.42))
        # Lower reservoir interface line (green)
        painter.setPen(QPen(QColor("#16a34a"), 1.5))
        painter.drawLine(QPointF(s*0.1, s*0.55), QPointF(s*0.9, s*0.55))
        # Permeable base
        painter.setPen(QPen(QColor("#0284c7"), 1.0))
        painter.setBrush(QBrush(QColor("#e0f2fe")))
        painter.drawRect(QRectF(s*0.1, s*0.6, s*0.8, s*0.25))

    elif icon_type in ("stress", "insitu"):
        # In-situ Stress Tensor (Overburden Sigma_v and Horizontal Sigma_h)
        painter.setPen(QPen(QColor("#475569"), 1.2))
        painter.drawRect(QRectF(s*0.25, s*0.25, s*0.5, s*0.5))
        # Sigma_v downward arrow
        painter.setPen(QPen(QColor("#dc2626"), 1.5))
        painter.drawLine(QPointF(s*0.5, s*0.08), QPointF(s*0.5, s*0.25))
        # Sigma_h inward horizontal arrows
        painter.setPen(QPen(QColor("#2563eb"), 1.5))
        painter.drawLine(QPointF(s*0.08, s*0.5), QPointF(s*0.25, s*0.5))
        painter.drawLine(QPointF(s*0.92, s*0.5), QPointF(s*0.75, s*0.5))

    elif icon_type == "uic":
        # EPA Class VI Safety Shield
        painter.setPen(QPen(QColor("#b91c1c"), 1.2))
        painter.setBrush(QBrush(QColor("#fee2e2")))
        shield = QPolygonF([
            QPointF(s*0.5, s*0.1),
            QPointF(s*0.85, s*0.25),
            QPointF(s*0.8, s*0.65),
            QPointF(s*0.5, s*0.9),
            QPointF(s*0.2, s*0.65),
            QPointF(s*0.15, s*0.25)
        ])
        painter.drawPolygon(shield)

    elif icon_type in ("audit", "quality"):
        # Blue Shield with Green Checkmark
        painter.setPen(QPen(QColor("#0284c7"), 1.2))
        painter.setBrush(QBrush(QColor("#e0f2fe")))
        shield = QPolygonF([
            QPointF(s*0.5, s*0.1),
            QPointF(s*0.85, s*0.25),
            QPointF(s*0.8, s*0.65),
            QPointF(s*0.5, s*0.9),
            QPointF(s*0.2, s*0.65),
            QPointF(s*0.15, s*0.25)
        ])
        painter.drawPolygon(shield)
        # Checkmark
        painter.setPen(QPen(QColor("#16a34a"), 1.8))
        painter.drawLine(QPointF(s*0.32, s*0.5), QPointF(s*0.45, s*0.68))
        painter.drawLine(QPointF(s*0.45, s*0.68), QPointF(s*0.7, s*0.35))

    elif icon_type in ("workstation", "surveillance"):
        # Multi-screen Diagnostic Dashboard
        painter.setPen(QPen(QColor("#475569"), 1.2))
        painter.setBrush(QBrush(QColor("#f8fafc")))
        painter.drawRoundedRect(QRectF(s*0.1, s*0.15, s*0.8, s*0.6), 2, 2)
        # 4 mini-screen quadrants
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRect(QRectF(s*0.18, s*0.23, s*0.28, s*0.18))
        painter.setPen(QPen(QColor("#4ade80"), 1.0))
        painter.drawRect(QRectF(s*0.54, s*0.23, s*0.28, s*0.18))
        painter.setPen(QPen(QColor("#f43f5e"), 1.0))
        painter.drawRect(QRectF(s*0.18, s*0.48, s*0.28, s*0.18))
        painter.setPen(QPen(QColor("#fbbf24"), 1.0))
        painter.drawRect(QRectF(s*0.54, s*0.48, s*0.28, s*0.18))
        # Stand
        painter.setPen(QPen(QColor("#475569"), 1.2))
        painter.drawLine(QPointF(s*0.5, s*0.75), QPointF(s*0.5, s*0.88))
        painter.drawLine(QPointF(s*0.35, s*0.88), QPointF(s*0.65, s*0.88))

    elif icon_type in ("table", "sheet"):
        # Spreadsheet table with header and grid
        painter.setPen(QPen(QColor("#0284c7"), 1.2))
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawRoundedRect(QRectF(s*0.12, s*0.12, s*0.76, s*0.76), 2, 2)
        # Header bar
        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.drawRect(QRectF(s*0.12, s*0.12, s*0.76, s*0.22))
        # Grid lines
        painter.setPen(QPen(QColor("#94a3b8"), 1.0))
        painter.drawLine(QPointF(s*0.12, s*0.56), QPointF(s*0.88, s*0.56))
        painter.drawLine(QPointF(s*0.50, s*0.34), QPointF(s*0.50, s*0.88))

    elif icon_type in ("graph", "chart", "plot"):
        # Coordinate axes with line chart
        painter.setPen(QPen(QColor("#475569"), 1.2))
        painter.drawLine(QPointF(s*0.15, s*0.15), QPointF(s*0.15, s*0.85))
        painter.drawLine(QPointF(s*0.15, s*0.85), QPointF(s*0.88, s*0.85))
        # Trend line
        painter.setPen(QPen(QColor("#0d6efd"), 1.6))
        painter.drawLine(QPointF(s*0.18, s*0.75), QPointF(s*0.42, s*0.45))
        painter.drawLine(QPointF(s*0.42, s*0.45), QPointF(s*0.65, s*0.58))
        painter.drawLine(QPointF(s*0.65, s*0.58), QPointF(s*0.85, s*0.22))

    elif icon_type in ("storage", "ccus", "sequestration"):
        # Underground Carbon Storage Tank with Downward Sequestration Arrow
        painter.setPen(QPen(QColor("#059669"), 1.2))
        painter.setBrush(QBrush(QColor("#d1fae5")))
        painter.drawRoundedRect(QRectF(s*0.15, s*0.2, s*0.7, s*0.6), 2, 2)
        # Downward Injection Arrow
        painter.setPen(QPen(QColor("#047857"), 1.6))
        painter.drawLine(QPointF(s*0.5, s*0.28), QPointF(s*0.5, s*0.65))
        painter.drawLine(QPointF(s*0.35, s*0.52), QPointF(s*0.5, s*0.68))
        painter.drawLine(QPointF(s*0.65, s*0.52), QPointF(s*0.5, s*0.68))

    elif icon_type in ("geothermal", "thermal", "energy"):
        # Geothermal Heat Waves & Thermal Energy
        painter.setPen(QPen(QColor("#ea580c"), 1.2))
        painter.setBrush(QBrush(QColor("#ffedd5")))
        # Base rock formation
        painter.drawRect(QRectF(s*0.12, s*0.68, s*0.76, s*0.2))
        # Rising heat waves
        painter.setPen(QPen(QColor("#dc2626"), 1.5))
        painter.drawArc(QRectF(s*0.2, s*0.25, s*0.2, s*0.35), 0, 180 * 16)
        painter.drawArc(QRectF(s*0.4, s*0.18, s*0.2, s*0.42), 0, 180 * 16)
        painter.drawArc(QRectF(s*0.6, s*0.25, s*0.2, s*0.35), 0, 180 * 16)

    elif icon_type in ("schedule", "gantt", "timeline"):
        # Timeline Gantt chart bars
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawLine(QPointF(s*0.15, s*0.15), QPointF(s*0.15, s*0.85))
        painter.drawLine(QPointF(s*0.15, s*0.85), QPointF(s*0.88, s*0.85))
        # Task bars
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.drawRoundedRect(QRectF(s*0.22, s*0.24, s*0.38, s*0.14), 2, 2)
        painter.setBrush(QBrush(QColor("#16a34a")))
        painter.drawRoundedRect(QRectF(s*0.45, s*0.44, s*0.38, s*0.14), 2, 2)
        painter.setBrush(QBrush(QColor("#f59e0b")))
        painter.drawRoundedRect(QRectF(s*0.32, s*0.64, s*0.42, s*0.14), 2, 2)

    else:
        # Default Bullet Dot
        painter.setPen(QPen(QColor("#64748b"), 1.0))
        painter.setBrush(QBrush(QColor("#94a3b8")))
        painter.drawEllipse(QRectF(s*0.3, s*0.3, s*0.4, s*0.4))

    painter.end()
    return QIcon(pixmap)
