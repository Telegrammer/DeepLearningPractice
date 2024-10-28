import sys

import torch
from PyQt5.QtWidgets import QApplication

from categorization.NumberIdentifier import TorchLinearNetwork, TorchTeacher, DrawNotepadHandler


def window(model, teacher):
    app = QApplication(sys.argv)
    w = DrawNotepadHandler(model, teacher)
    sys.exit(app.exec_())


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = TorchLinearNetwork((784, 20, 10)).to(device)
    teacher = TorchTeacher(epochs=5, device=device)

    try:
        param_model = torch.load('model_state_dict_epoch_3.pt')
        model.load_state_dict(param_model)
    except FileNotFoundError:
        print("Сеть не обучилась")
        teacher.teach_model(model)
    except RuntimeError:
        print("У сети новая топология")
        teacher.teach_model(model)
    window(model, teacher)
