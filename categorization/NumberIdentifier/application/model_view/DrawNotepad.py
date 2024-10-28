from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QColorDialog

from ..view.DrawNotepad_ui import Ui_Form

__all__ = ["DrawNotepad"]

# Класс DrawNotepad является заметкой графического типа
class DrawNotepad(QWidget, Ui_Form):

    def __init__(self, button_pressed):
        super().__init__()
        self.color = (0, 0, 0)
        self.setGeometry(0, 0,
                         QDesktopWidget().availableGeometry().width(),
                         QDesktopWidget().availableGeometry().height())
        self.setupUi(self)

        self.canvas = QPixmap(self.frameGeometry().width(), self.frameGeometry().height())
        self.canvas_heart.setPixmap(self.canvas)
        # self.buttonGroup.buttonClicked.connect(self.set_color)
        # self.colorSelectButton.clicked.connect(self.set_color_alt)
        self.spinBox.valueChanged.connect(lambda: self.canvas_heart.set_pen_width(self.spinBox.value()))
        self.backColorSelect.clicked.connect(self.save_number)
        self.button_pressed = button_pressed

    # close_event сохранет данные в папке pictures, а также в базе данных notes.db
    def closeEvent(self, event):
        pass

    # Функции, отвечающие за изменение цвета кисти и холста
    def set_color(self, button):
        self.color = button.styleSheet().split(';')[0]
        self.color = self.color[self.color.index('(') + 1:self.color.index(')')]
        self.color = [int(number) for number in self.color.split(', ')]
        self.color = QColor(self.color[0], self.color[1], self.color[2])
        self.canvas_heart.set_pen_color(self.color)

    def set_color_alt(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.canvas_heart.set_pen_color(color)

    def save_number(self):
        result = self.canvas_heart.pixmap()
        result = result.scaled(28, 28, Qt.KeepAspectRatio)
        result.save(r'pictures\{}.png'.format(self.windowTitle()))
        color = QColor("black")
        if color.isValid():
            self.canvas.fill(color)
            self.canvas_heart.setPixmap(self.canvas)
        self.button_pressed.signal.emit()
