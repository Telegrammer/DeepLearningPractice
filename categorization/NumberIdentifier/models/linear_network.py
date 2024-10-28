from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import torch

__all__ = ["AbstractLinearNetwork", "np", "deque", "torch"]


class AbstractLinearNetwork(ABC):
    def __init__(self, topology: tuple[int]):
        self.topology = topology

    @abstractmethod
    def forward(self, input_tensor) -> [torch.Tensor, np.ndarray]:
        pass

    # @abstractmethod
    # def back_propagation(self, expected_result: float, output: [torch.Tensor, np.ndarray]) -> \
    #         deque[torch.Tensor, np.ndarray]:
    #     pass
