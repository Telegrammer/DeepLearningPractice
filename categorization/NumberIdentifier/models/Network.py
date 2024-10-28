import enum

from .linear_network import *
from .functions import *

__all__ = ["Network"]


class LayerType(enum.IntEnum):
    input = 0
    output = -1


def to_full(y, num_classes):
    y_full = np.zeros((num_classes, 1))
    y_full[y] = 1
    return y_full


class Network(AbstractLinearNetwork):
    def __init__(self, topology: tuple[int] = (784, 200, 10),
                 layer_neurons_params: tuple[tuple[float]] = ((-1, 1), (-1, 1)),
                 hidden_activation_function: object = sigmoid,
                 activation_function: object = softmax) -> object:

        super().__init__(topology)

        self.__weights = list()
        self.__biases = list()

        self.__hidden_activation_function = hidden_activation_function
        self.__activation_function = activation_function

        for i in range(1, len(topology)):
            borders = layer_neurons_params[i - 1]
            self.__weights.append(np.random.uniform(borders[0], borders[1], (topology[i], topology[i - 1])))
            self.__biases.append(np.zeros([topology[i], 1]))

        self.__inner_values: deque[np.ndarray] = list()

    def forward(self, predict: np.ndarray) -> np.ndarray:
        predict: np.ndarray = predict.reshape([self.topology[LayerType.input], 1]) / 255
        self.__inner_values.append(predict[:])
        for i in range(0, len(self.topology) - 2):
            predict = (self.__weights[i] @ predict) + self.__biases[i]
            predict.clip(-20, 20)
            predict = self.__hidden_activation_function(predict)
            self.__inner_values.append(predict[:])
        predict = self.__activation_function(
            (self.__weights[-1] @ self.__hidden_activation_function(self.__inner_values[-1])) + self.__biases[-1])

        return predict

    def update(self, alpha: float, gradient):
        for i in range(len(self.topology) - 1):
            self.__weights[i] -= alpha * gradient[-1 - i][0]
            self.__biases[i] -= alpha * gradient[-1 - i][1]
        pass

    def back_propagation(self, expected_result: float, output: np.ndarray):
        gradient = deque()
        y_full = to_full(expected_result, self.topology[-1])
        dE_dt = output - y_full
        for i in range(0, len(self.topology) - 1):
            inner_value = self.__inner_values.pop()
            dE_dW = dE_dt @ inner_value.T
            dE_db = dE_dt
            dE_dh = self.__weights[-1 - i].T @ dE_dt
            if self.__hidden_activation_function == sigmoid:
                dE_dt = dE_dh * sigmoid_deriv(inner_value)
            elif self.__hidden_activation_function == ReLU:
                dE_dt = dE_dh * relu_deriv(inner_value)
            gradient.append((dE_dW, dE_db))

        return gradient
