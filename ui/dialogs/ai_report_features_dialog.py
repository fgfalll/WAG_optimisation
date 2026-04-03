from PyQt6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QDialogButtonBox
from PyQt6.QtCore import Qt

class AIReportFeaturesDialog(QDialog):
    def __init__(self, available_sections, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Define AI Report Features")
        
        layout = QVBoxLayout(self)
        
        self.list_widget = QListWidget()
        for key, name in available_sections.items():
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.list_widget.addItem(item)
            
        layout.addWidget(self.list_widget)
        
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_selected_features(self):
        features = {}
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                key = item.data(Qt.ItemDataRole.UserRole)
                features[key] = True
        return features
