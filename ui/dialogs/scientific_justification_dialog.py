from PyQt6.QtWidgets import QWidget, QHBoxLayout, QListWidget, QStackedWidget

from ui.dialogs.overview_dialog import OverviewWidget
from ui.dialogs.recovery_models_dialog import RecoveryModelsWidget
from ui.dialogs.eos_modeling_dialog import EOSModelingWidget
from ui.dialogs.profiler_dialog import ProfilerWidget
from ui.dialogs.breakthrough_physics_dialog import BreakthroughPhysicsWidget

class ScientificJustificationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Scientific Justification"))
        self.setMinimumSize(800, 600)

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)

        # Side Menu
        self.side_menu = QListWidget()
        self.side_menu.setFixedWidth(200)
        main_layout.addWidget(self.side_menu)

        # Content Area
        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.content_stack)

        # Add sections
        self.add_section("Overview", OverviewWidget())
        self.add_section("Recovery Models", RecoveryModelsWidget())
        self.add_section("EOS Modeling", EOSModelingWidget())
        self.add_section("Profiler", ProfilerWidget())
        self.add_section("Breakthrough Physics", BreakthroughPhysicsWidget())

        self.side_menu.currentRowChanged.connect(self.content_stack.setCurrentIndex)

    def add_section(self, name, widget):
        self.side_menu.addItem(name)
        self.content_stack.addWidget(widget)