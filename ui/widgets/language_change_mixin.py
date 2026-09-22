from PyQt6.QtCore import QEvent


class LanguageChangeMixin:
    """Mixin class to handle language change events for widgets with retranslateUi."""

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.LanguageChange:
            self.retranslateUi()
        return super(LanguageChangeMixin, self).changeEvent(event)