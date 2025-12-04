import numpy as np

class SimpleNN:
    def __init__(self, input_size=4, hidden_size=8, output_size=3):
        # Random initialization
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        