from __future__ import annotations
from typing import Optional, Dict, List, Tuple

from PyQt6.QtWidgets import QTreeView, QAbstractItemView, QWidget
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt

class ParameterSelectionTreeView(QTreeView):
    """A specialized QTreeView for selecting parameters with check boxes."""
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.model = QStandardItemModel(self)
        self.setModel(self.model)
        self.setHeaderHidden(True)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

    def populate(self, param_structure: Dict[str, List[Tuple[str, str]]]):
        self.model.clear()
        root = self.model.invisibleRootItem()
        for category, params in param_structure.items():
            cat_item = QStandardItem(category)
            cat_item.setEditable(False)
            cat_item.setSelectable(False)
            for path, name in params:
                param_item = QStandardItem(name)
                param_item.setData(path, Qt.ItemDataRole.UserRole)
                param_item.setCheckable(True)
                param_item.setEditable(False)
                cat_item.appendRow(param_item)
            root.appendRow(cat_item)
        self.expandAll()

    def get_selected_paths(self) -> List[str]:
        paths = []
        root = self.model.invisibleRootItem()
        for i in range(root.rowCount()):
            cat_item = root.child(i)
            if cat_item is None: continue
            for j in range(cat_item.rowCount()):
                param_item = cat_item.child(j)
                if param_item is None: continue
                if param_item.checkState() == Qt.CheckState.Checked:
                    paths.append(param_item.data(Qt.ItemDataRole.UserRole))
        return paths

    def get_selected_items(self) -> List[Tuple[str, str]]:
        """Returns a list of (path, name) tuples for all checked parameters."""
        items = []
        root = self.model.invisibleRootItem()
        for i in range(root.rowCount()):
            cat_item = root.child(i)
            if cat_item is None: continue
            for j in range(cat_item.rowCount()):
                param_item = cat_item.child(j)
                if param_item is None: continue
                if param_item.checkState() == Qt.CheckState.Checked:
                    path = param_item.data(Qt.ItemDataRole.UserRole)
                    name = param_item.text()
                    items.append((path, name))
        return items

    def clear_selection(self):
        """Unchecks all checkable items in the tree."""
        self.model.blockSignals(True)
        try:
            root = self.model.invisibleRootItem()
            for i in range(root.rowCount()):
                cat_item = root.child(i)
                if cat_item is None: continue
                for j in range(cat_item.rowCount()):
                    param_item = cat_item.child(j)
                    if param_item and param_item.isCheckable():
                        param_item.setCheckState(Qt.CheckState.Unchecked)
        finally:
            self.model.blockSignals(False)
