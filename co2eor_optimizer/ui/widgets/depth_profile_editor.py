import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPen, QBrush, QPolygonF, QFont, QColor
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QPoint

class DepthProfileEditor(QWidget):
    """
    An interactive widget to graphically define a 2D well path with axes and tooltips.
    - Endpoints (Top/Bottom) are fixed vertically.
    - New points are added by clicking on a path segment.
    - Intermediate points are constrained vertically between their neighbors.
    - Intermediate points can be deleted with a double-click.
    """
    pathChanged = pyqtSignal(list)
    MARGIN = 50 # pixels for axis labels

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self.path_points: list[QPointF] = []
        self._dragging_point_idx: int | None = None
        self._hover_point_idx: int | None = None
        self._hover_point_data: QPointF | None = None
        self.setMouseTracking(True)

        # Visual styles
        self.axis_pen = QPen(Qt.GlobalColor.black, 1)
        self.axis_font = QFont("Segoe UI", 8)
        self.path_pen = QPen(Qt.GlobalColor.blue, 2)
        self.point_brush = QBrush(Qt.GlobalColor.blue)
        self.hover_brush = QBrush(Qt.GlobalColor.red)
        self.tooltip_brush = QBrush(QColor(0, 0, 0, 180))
        self.tooltip_pen = QPen(Qt.GlobalColor.white)

    def set_path(self, path: list[QPointF]):
        """
        Sets the entire well path from an external list of points.
        This is used to initialize the editor with existing data.
        """
        self.path_points = [QPointF(p) for p in path]
        self.path_points = self.get_path()
        self.update()

    def set_path_endpoints(self, top_md: float, bottom_md: float):
        if not self.path_points:
            self.path_points = [QPointF(0, top_md), QPointF(0, bottom_md)]
        else:
            self.path_points[0].setY(top_md)
            self.path_points[-1].setY(bottom_md)
        self.path_points = self.get_path()
        self.update()
        self.pathChanged.emit(self.path_points)

    def get_path(self) -> list[QPointF]:
        return sorted(self.path_points, key=lambda p: p.y())

    def _get_logical_bounds(self) -> tuple[float, float, float, float]:
        """
        Calculates the logical coordinate bounds for the plot view, ensuring a
        non-zero viewport size even for vertical or horizontal lines.
        """
        all_points = self.get_path()
        if not all_points:
            return -50.0, 50.0, 0.0, 1000.0

        x_coords = [p.x() for p in all_points]
        y_coords = [p.y() for p in all_points]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # FIX: Explicitly handle zero-range cases by applying a fixed pad.
        x_range = max_x - min_x
        y_range = max_y - min_y
        
        x_pad = 50.0 if x_range < 1e-6 else x_range * 0.15
        y_pad = 50.0 if y_range < 1e-6 else y_range * 0.1

        return min_x - x_pad, max_x + x_pad, min_y - y_pad, max_y + y_pad

    def _transform_to_widget(self, point: QPointF) -> QPointF:
        min_x, max_x, min_y, max_y = self._get_logical_bounds()
        logical_width = max_x - min_x
        logical_height = max_y - min_y
        
        if logical_width < 1e-6 or logical_height < 1e-6:
             return QPointF(self.width() / 2, self.height() / 2)

        plot_width = self.width() - self.MARGIN
        plot_height = self.height() - self.MARGIN
        
        x = self.MARGIN + ((point.x() - min_x) / logical_width) * plot_width
        y = ((point.y() - min_y) / logical_height) * plot_height
        return QPointF(x, y)

    def _transform_from_widget(self, point: QPointF) -> QPointF:
        min_x, max_x, min_y, max_y = self._get_logical_bounds()
        logical_width = max_x - min_x
        logical_height = max_y - min_y
        if logical_width < 1e-6 or logical_height < 1e-6: return QPointF(0,0)
        
        plot_width = self.width() - self.MARGIN
        plot_height = self.height() - self.MARGIN
        logical_x = min_x + ((point.x() - self.MARGIN) / plot_width) * logical_width
        logical_y = min_y + (point.y() / plot_height) * logical_height
        return QPointF(logical_x, logical_y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#FEFEFE"))

        # FIX: Guard against painting before the widget is properly sized.
        if not self.path_points or self.width() <= self.MARGIN or self.height() <= self.MARGIN:
            return

        self._draw_axes(painter)
        
        painter.setPen(self.path_pen)
        widget_path = QPolygonF([self._transform_to_widget(p) for p in self.get_path()])
        painter.drawPolyline(widget_path)

        for i, point in enumerate(self.get_path()):
            widget_point = self._transform_to_widget(point)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.hover_brush if i == self._hover_point_idx else self.point_brush)
            painter.drawEllipse(widget_point, 6, 6)

        self._draw_hover_tooltip(painter)

    def _draw_axes(self, painter: QPainter):
        painter.setPen(self.axis_pen)
        painter.setFont(self.axis_font)
        min_x, max_x, min_y, max_y = self._get_logical_bounds()
        
        # Y-Axis (MD)
        painter.drawLine(self.MARGIN, 0, self.MARGIN, self.height() - self.MARGIN)
        painter.drawText(QRectF(0, 0, self.MARGIN - 5, 20), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, f"{self.get_path()[0].y():.0f} ft")
        painter.drawText(QRectF(0, self.height() - self.MARGIN - 20, self.MARGIN - 5, 20), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, f"{self.get_path()[-1].y():.0f} ft")
        
        # X-Axis (Deviation)
        painter.drawLine(self.MARGIN, self.height() - self.MARGIN, self.width(), self.height() - self.MARGIN)
        painter.drawText(QRectF(self.MARGIN, self.height() - self.MARGIN + 5, 50, 15), Qt.AlignmentFlag.AlignLeft, f"{min_x:.1f}")
        painter.drawText(QRectF(self.width() - 60, self.height() - self.MARGIN + 5, 55, 15), Qt.AlignmentFlag.AlignRight, f"{max_x:.1f}")

    def _draw_hover_tooltip(self, painter: QPainter):
        if self._hover_point_idx is None or self._hover_point_data is None: return

        widget_point = self._transform_to_widget(self._hover_point_data)
        text = f"MD: {self._hover_point_data.y():.1f}\nDev: {self._hover_point_data.x():.1f}"
        
        rect = QRectF(0, 0, 110, 40)
        rect.moveCenter(widget_point + QPointF(0, -35))
        if rect.left() < 0: rect.setLeft(5)
        if rect.right() > self.width(): rect.setRight(self.width() - 5)
        
        painter.setBrush(self.tooltip_brush)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 5, 5)
        painter.setPen(self.tooltip_pen)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _get_point_at(self, pos: QPoint, tolerance: int = 12) -> int | None:
        pos_f = QPointF(pos)
        for i, p in enumerate(self.get_path()):
            if (pos_f - self._transform_to_widget(p)).manhattanLength() < tolerance: return i
        return None

    def _get_segment_at(self, pos: QPointF) -> int | None:
        min_dist_sq = 15**2
        closest_segment_idx = None
        
        path = self.get_path()
        for i in range(len(path) - 1):
            p1 = self._transform_to_widget(path[i])
            p2 = self._transform_to_widget(path[i+1])
            line_vec = p2 - p1
            point_vec = pos - p1
            line_len_sq = line_vec.x()**2 + line_vec.y()**2
            if line_len_sq < 1e-6: continue
            t = max(0, min(1, QPointF.dotProduct(point_vec, line_vec) / line_len_sq))
            projection = p1 + t * line_vec
            dist_sq = (pos - projection).x()**2 + (pos - projection).y()**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_segment_idx = i
        return closest_segment_idx

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton: return
        
        idx = self._get_point_at(event.pos())
        if idx is not None:
            self._dragging_point_idx = idx
        else:
            segment_idx = self._get_segment_at(QPointF(event.pos()))
            if segment_idx is not None:
                new_logical_point = self._transform_from_widget(QPointF(event.pos()))
                self.path_points.insert(segment_idx + 1, new_logical_point)
                self.path_points = self.get_path()
                self._dragging_point_idx = segment_idx + 1
        self.update()

    def mouseMoveEvent(self, event):
        if self._dragging_point_idx is not None:
            new_pos = self._transform_from_widget(QPointF(event.pos()))
            path = self.get_path()
            dragged_point = path[self._dragging_point_idx]
            if self._dragging_point_idx == 0 or self._dragging_point_idx == len(path) - 1:
                dragged_point.setX(new_pos.x())
            else:
                prev_point = path[self._dragging_point_idx - 1]
                next_point = path[self._dragging_point_idx + 1]
                clamped_y = max(prev_point.y(), min(new_pos.y(), next_point.y()))
                dragged_point.setX(new_pos.x())
                dragged_point.setY(clamped_y)
            self.update()
        else:
            self._hover_point_idx = self._get_point_at(event.pos())
            self._hover_point_data = self.get_path()[self._hover_point_idx] if self._hover_point_idx is not None else None
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging_point_idx is not None:
            self._dragging_point_idx = None
            self.path_points = self.get_path()
            self.pathChanged.emit(self.path_points)
            self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton: return
        
        idx_to_delete = self._get_point_at(event.pos())
        if idx_to_delete is not None and 0 < idx_to_delete < len(self.get_path()) - 1:
            del self.path_points[idx_to_delete]
            self.path_points = self.get_path()
            self.pathChanged.emit(self.path_points)
            self._hover_point_idx = None
            self._hover_point_data = None
            self.update()