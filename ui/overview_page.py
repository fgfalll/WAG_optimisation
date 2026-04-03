import logging
import os
from datetime import datetime
from typing import Optional, List

from PyQt6.QtCore import (
    Qt, pyqtSignal, QSize, QSettings, QEvent,
    QPropertyAnimation, QEasingCurve, QPoint
)
from PyQt6.QtGui import (
    QIcon, QFont, QPainter, QFontDatabase,
    QPixmap, QPainterPath, QColor, QLinearGradient, QBrush
)
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QApplication, QHBoxLayout, QFrame, QScrollArea, QSplitter
)

logger = logging.getLogger(__name__)

# --- SVG Icon Data ---
# Storing SVG data for clean, portable icons.
# Icons from feathericons.com (MIT License)
SVG_ICONS = {
    "new_project": '''
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"
         fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"
         stroke-linejoin="round" class="feather feather-plus-square">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
        <line x1="12" y1="8" x2="12" y2="16"></line>
        <line x1="8" y1="12" x2="16" y2="12"></line>
    </svg>''',
    "load_project": '''
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"
         fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"
         stroke-linejoin="round" class="feather feather-folder">
        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
    </svg>''',
    "quick_start": '''
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"
         fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"
         stroke-linejoin="round" class="feather feather-play-circle">
        <circle cx="12" cy="12" r="10"></circle>
        <polygon points="10 8 16 12 10 16 10 8"></polygon>
    </svg>''',
    "delete": '''
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"
         fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"
         stroke-linejoin="round" class="feather feather-x-circle">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="15" y1="9" x2="9" y2="15"></line>
        <line x1="9" y1="9" x2="15" y2="15"></line>
    </svg>'''
}

def create_icon_from_svg(svg_data: str, color: str) -> QIcon:
    """Renders an SVG string into a QIcon with a specified color."""
    svg_data = svg_data.replace('stroke="currentColor"', f'stroke="{color}"')
    svg_renderer = QSvgRenderer(svg_data.encode('utf-8'))
    pixmap = QPixmap(svg_renderer.defaultSize())
    pixmap.fill(Qt.GlobalColor.transparent)
    
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    svg_renderer.render(painter)
    painter.end()
    
    return QIcon(pixmap)


class AnimatedButton(QPushButton):
    """A QPushButton with a subtle animation on hover for a more dynamic feel."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._animation = QPropertyAnimation(self, b"geometry")
        self._animation.setDuration(150)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._original_geometry = self.geometry()

    def enterEvent(self, event):
        self._original_geometry = self.geometry()
        self._animation.setStartValue(self._original_geometry)
        # Animate by growing slightly
        end_geometry = self._original_geometry.adjusted(-2, -2, 2, 2)
        self._animation.setEndValue(end_geometry)
        self._animation.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        # Animate back to original size
        self._animation.setStartValue(self.geometry())
        self._animation.setEndValue(self._original_geometry)
        self._animation.start()
        super().leaveEvent(event)


class OverviewPageWidget(QWidget):
    """
    The initial welcome screen for the application, styled with a modern dark theme.
    It provides top-level actions and a list of recent projects.
    """
    start_new_project_requested = pyqtSignal()
    load_existing_project_requested = pyqtSignal()
    quick_start_requested = pyqtSignal()
    open_recent_project_requested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None, preferences_manager=None):
        super().__init__(parent)
        self.setObjectName("OverviewPage")
        self.settings = QSettings("CO2EORSuite", "Overview")
        self.preferences_manager = preferences_manager
        self._setup_ui()
        self._connect_signals()
        self._apply_styling()
        self.retranslateUi()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#ffffff"))

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(50, 40, 50, 40)
        main_layout.setSpacing(40)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        header_widget = self._create_header_widget()
        main_layout.addWidget(header_widget)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(40)

        left_column_widget = self._create_actions_section()
        right_column_widget = self._create_recent_projects_section()

        content_layout.addWidget(left_column_widget, 1)
        content_layout.addWidget(right_column_widget, 1)

        main_layout.addLayout(content_layout)

    def _create_header_widget(self) -> QWidget:
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 20)
        header_layout.setSpacing(8)
        header_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_label = QLabel()
        self.title_label.setObjectName("TitleLabel")
        
        self.subtitle_label = QLabel()
        self.subtitle_label.setObjectName("SubtitleLabel")

        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.subtitle_label)

        return header_widget

    def _create_actions_section(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("CardFrame")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(25, 20, 25, 25)
        layout.setSpacing(20)

        title = QLabel(self.tr("Get Started"))
        title.setObjectName("SectionTitle")
        layout.addWidget(title)

        self.start_new_button = self._create_action_button(
            object_name="start_newButton",
            icon=create_icon_from_svg(SVG_ICONS["new_project"], "#000000"),
        )
        self.load_existing_button = self._create_action_button(
            object_name="load_existingButton",
            icon=create_icon_from_svg(SVG_ICONS["load_project"], "#000000"),
        )
        self.quick_start_button = self._create_action_button(
            object_name="quick_startButton",
            icon=create_icon_from_svg(SVG_ICONS["quick_start"], "#000000"),
        )

        layout.addWidget(self.start_new_button)
        layout.addWidget(self.load_existing_button)
        layout.addWidget(self.quick_start_button)
        layout.addStretch()

        return frame

    def _create_action_button(self, object_name: str, icon: QIcon) -> AnimatedButton:
        button = AnimatedButton(icon, "")
        button.setObjectName(object_name)
        button.setMinimumHeight(60)
        button.setIconSize(QSize(28, 28))
        return button

    def _create_recent_projects_section(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("CardFrame")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(25, 20, 25, 25)
        layout.setSpacing(15)

        self.recent_projects_title = QLabel()
        self.recent_projects_title.setObjectName("SectionTitle")
        layout.addWidget(self.recent_projects_title)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setObjectName("RecentProjectsScroll")

        self.recent_projects_widget = QWidget()
        self.recent_projects_layout = QVBoxLayout(self.recent_projects_widget)
        self.recent_projects_layout.setSpacing(15)
        self.recent_projects_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.no_projects_label = QLabel()
        self.no_projects_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_projects_label.setObjectName("NoProjectsLabel")
        self.no_projects_label.setWordWrap(True)
        self.no_projects_label.hide()
        self.recent_projects_layout.addWidget(self.no_projects_label)

        scroll_area.setWidget(self.recent_projects_widget)
        layout.addWidget(scroll_area)

        self._load_recent_projects()
        return frame

    def _load_recent_projects(self):
        self._clear_recent_projects_list(keep_placeholder=True)
        recent_projects = self.settings.value("recent_projects", [], type=list)

        if not recent_projects:
            self.no_projects_label.show()
            return

        self.no_projects_label.hide()
        max_files = (
            self.preferences_manager.general.max_recent_files
            if self.preferences_manager else 7
        )
        for project_path in recent_projects[:max_files]:  # Limit to max_recent_files
            if os.path.exists(project_path):
                self._add_recent_project_item(project_path)
    
    def _add_recent_project_item(self, project_path: str):
        project_button = QPushButton()
        project_button.setToolTip(project_path)
        project_button.setObjectName("RecentProjectButton")
        project_button.setCursor(Qt.CursorShape.PointingHandCursor)
        project_button.clicked.connect(
            lambda: self.open_recent_project_requested.emit(project_path)
        )
        project_button.setMinimumHeight(80)

        main_layout = QHBoxLayout(project_button)
        main_layout.setContentsMargins(15, 12, 15, 12)
        main_layout.setSpacing(15)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(3)

        project_name = os.path.basename(project_path)
        modified_time = datetime.fromtimestamp(os.path.getmtime(project_path))
        time_str = self.tr("Modified: %1").replace(
            "%1", modified_time.strftime("%b %d, %Y %H:%M")
        )

        name_label = QLabel(project_name)
        name_label.setObjectName("ProjectNameLabel")

        path_label = QLabel(project_path)
        path_label.setObjectName("ProjectPathLabel")
        
        info_layout.addWidget(name_label)
        info_layout.addWidget(path_label)
        info_layout.addStretch()
        info_layout.addWidget(QLabel(time_str))

        delete_button = QPushButton()
        delete_button.setObjectName("DeleteButton")
        delete_button.setIcon(create_icon_from_svg(SVG_ICONS["delete"], "#c0392b"))
        delete_button.setFixedSize(32, 32)
        delete_button.setIconSize(QSize(24, 24))
        delete_button.setCursor(Qt.CursorShape.PointingHandCursor)
        delete_button.clicked.connect(lambda: self._handle_delete_recent_project(project_path))

        main_layout.addLayout(info_layout)
        main_layout.addStretch()
        main_layout.addWidget(delete_button)

        self.recent_projects_layout.addWidget(project_button)

    def _handle_delete_recent_project(self, project_path: str):
        recent_projects = self.get_recent_projects()
        if project_path in recent_projects:
            recent_projects.remove(project_path)
            self.set_recent_projects(recent_projects)

    def _connect_signals(self):
        self.start_new_button.clicked.connect(self.start_new_project_requested)
        self.load_existing_button.clicked.connect(self.load_existing_project_requested)
        self.quick_start_button.clicked.connect(self.quick_start_requested)


    def _apply_styling(self):
        self.setStyleSheet('''
            #OverviewPage {
                font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', sans-serif;
            }
            #TitleLabel {
                color: #2c3e50;
                font-size: 48px;
                font-weight: 700;
            }
            #SubtitleLabel {
                color: #7f8c8d;
                font-size: 20px;
                font-weight: 300;
            }
            #CardFrame {
                background-color: #ecf0f1;
                border-radius: 12px;
                border: 1px solid #bdc3c7;
            }
            #SectionTitle {
                color: #2c3e50;
                font-size: 24px;
                font-weight: 600;
                padding-bottom: 10px;
                border-bottom: 1px solid #bdc3c7;
            }
            AnimatedButton {
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
                font-weight: 600;
                text-align: center;
                color: white;
                background-color: #3498db;
                border: none;
            }
            AnimatedButton:hover {
                background-color: #2980b9;
            }
            #start_newButton {
                background-color: #2ecc71;
            }
            #start_newButton:hover {
                background-color: #27ae60;
            }
            #load_existingButton {
                background-color: #f1c40f;
            }
            #load_existingButton:hover {
                background-color: #f39c12;
            }
            #quick_startButton {
                background-color: #9b59b6;
            }
            #quick_startButton:hover {
                background-color: #8e44ad;
            }

            QScrollArea {
                border: none;
                background: transparent;
            }
            QWidget#RecentProjectsScroll > QWidget {
                background: transparent;
            }

            #RecentProjectButton {
                text-align: left;
                background-color: #ffffff;
                border: 1px solid #bdc3c7;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 10px;
            }
            #RecentProjectButton:hover {
                background-color: #f8f9f9;
            }
            #ProjectNameLabel {
                font-size: 16px;
                font-weight: 600;
                color: #2c3e50;
            }
            #ProjectPathLabel {
                font-size: 12px;
                color: #7f8c8d;
            }
            #ProjectTimeLabel {
                color: #7f8c8d;
                font-size: 12px;
            }
            #DeleteButton {
                background-color: transparent;
                border: none;
            }
            #DeleteButton:hover {
                background-color: #e74c3c;
            }

            #NoProjectsLabel {
                color: #7f8c8d;
                font-style: italic;
                padding: 40px 20px;
                background-color: transparent;
                border-radius: 8px;
                border: 1px dashed #bdc3c7;
            }
        ''')

    def retranslateUi(self):
        app_name = "CO2 Optimiser PhD project"
        version = QApplication.applicationVersion() or "0.8.5-alpha"
        self.title_label.setText(app_name)
        self.subtitle_label.setText(
            self.tr("Professional-grade CO₂ Enhanced Oil Recovery Analysis")
        )

        self.start_new_button.setText(self.tr("  Start New Project"))
        self.start_new_button.setToolTip(
            self.tr("Create a new project from data files")
        )
        
        self.load_existing_button.setText(self.tr("  Load Existing Project"))
        self.load_existing_button.setToolTip(
            self.tr("Open a previously saved project")
        )

        self.quick_start_button.setText(self.tr("  Quick Start Demo"))
        self.quick_start_button.setToolTip(
            self.tr("Launch with sample data")
        )

        self.recent_projects_title.setText(self.tr("Recent Projects"))
        self.no_projects_label.setText(
            self.tr("No recent projects found.\nStart a new project to get started!")
        )
        # Reload projects to update date format on language change
        self._load_recent_projects()

    def get_recent_projects(self) -> List[str]:
        """Returns the current list of recent project paths from settings."""
        return self.settings.value("recent_projects", [], type=list)

    def set_recent_projects(self, projects: List[str]):
        """Sets the list of recent projects, replacing the existing one."""
        max_files = (
            self.preferences_manager.general.max_recent_files
            if self.preferences_manager else 10
        )
        self.settings.setValue("recent_projects", projects[:max_files])
        self._load_recent_projects()

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        super().changeEvent(event)

    def add_recent_project(self, project_path: str):
        if not project_path or not os.path.exists(project_path):
            # If path is empty or doesn't exist, just reload to prune dead links
            self._load_recent_projects()
            return
        
        recent_projects = self.get_recent_projects()
        
        if project_path in recent_projects:
            recent_projects.remove(project_path)
        recent_projects.insert(0, project_path)
        
        max_files = (
            self.preferences_manager.general.max_recent_files
            if self.preferences_manager else 10
        )
        self.settings.setValue("recent_projects", recent_projects[:max_files])
        self._load_recent_projects()
        
    def _clear_recent_projects_list(self, keep_placeholder: bool = False):
        for i in reversed(range(self.recent_projects_layout.count())):
            item = self.recent_projects_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if not (widget == self.no_projects_label and keep_placeholder):
                    widget.deleteLater()