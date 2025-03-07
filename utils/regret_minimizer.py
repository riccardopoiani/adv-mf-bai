import numpy as np


class GradientAscent:
    """
    Exponential weights algorithm
    """

    def __init__(self, dim: int, lr: float):
        self.dim = dim
        self.t = 0
        self.lr = lr
        self.g_t = np.zeros(dim, dtype=np.float128)

    def get_action(self) -> np.array:
        lr = self.get_lr()
        return np.exp(lr * (self.g_t - np.min(self.g_t))) / np.sum(np.exp(lr * (self.g_t - np.min(self.g_t))))
        # return np.exp(lr * self.g_t) / np.sum(np.exp(lr * self.g_t))

    def feed(self, gradient):
        self.t += 1
        self.g_t += gradient

    def get_lr(self):
        if self.lr is None:
            return np.sqrt(1 / (self.t + 1))
        return self.lr
