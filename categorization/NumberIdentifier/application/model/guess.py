import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as img

from ...models import Network, TorchLinearNetwork

__all__ = ["guess"]


def guess(network: [Network, TorchLinearNetwork], device: str):
    im = img.open("pictures/Form.png")
    im = im.convert("L")
    pix = im.load()
    g = np.empty(shape=(28, 28))

    for i in range(28):
        for j in range(28):
            g[i, j] = pix[i, j]

    g = g.T / 255
    g = torch.from_numpy(g).float()
    g = g.reshape(-1, 28 * 28).to(device)
    pred = network.forward(g)
    x_axis = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    y_axis = pred.softmax(dim=1).cpu().detach().numpy()[0]
    plt.clf()
    plt.bar(x_axis, y_axis)
    plt.show()