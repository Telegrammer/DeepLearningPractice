import Emitters
from .DrawNotepad import DrawNotepad
from ..model import guess

__all__ = ["DrawNotepadHandler"]


class DrawNotepadHandler:
    def __init__(self, network, teacher):
        self.__network = network
        self.__teacher = teacher

        self.button_pressed = Emitters.VoidEmitter()
        self.button_pressed.signal.connect(self.start_guess)

        self.__draw_notepad = DrawNotepad(self.button_pressed)
        self.__draw_notepad.show()

    def start_guess(self):
        guess(self.__network, self.__teacher.get_device())
