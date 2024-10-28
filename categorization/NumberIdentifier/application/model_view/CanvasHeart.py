from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QLabel

__all__ = ["CanvasHeart"]


# Класс CanvasHeart является скелетом для холста заметки класса DrawNotepad
class CanvasHeart(QLabel):
    def __init__(self, parent=None):
        super(CanvasHeart, self).__init__(parent)
        self.last_x, self.last_y = None, None
        self.pen_color = QColor('#ffffff')
        self.pen_width = 40
        self.setContentsMargins(0, 0, 0, 0)
        self.__painter = QPainter()

    # 2 Функции ниже отвечают за изменяемые характеристики кисти (Толщина, цвет)
    def set_pen_color(self, color):
        self.pen_color = QColor(color)

    def set_pen_width(self, width):
        self.pen_width = width

    # mouseMoveEvent отвечает за само рисование на холсте заметки
    def mouseMoveEvent(self, event):
        if self.last_x is None:
            self.last_x = event.x()
            self.last_y = event.y()
            return

        self.__painter.begin(self.pixmap())
        self.pen_color.setAlpha(50)
        self.pen_width += 20
        pen = QPen(self.pen_color, self.pen_width + 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.__painter.setPen(pen)
        self.__painter.drawLine(self.last_x, self.last_y + 6, event.x(), event.y() + 6)
        self.pen_color.setAlpha(255)
        self.pen_width -= 20
        pen = QPen(self.pen_color, self.pen_width + 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.__painter.setPen(pen)
        self.__painter.drawLine(self.last_x, self.last_y + 6, event.x(), event.y() + 6)
        self.__painter.end()
        self.update()

        self.last_x = event.x()
        self.last_y = event.y()

    def mouseReleaseEvent(self, event):
        self.last_x = None
        self.last_y = None
