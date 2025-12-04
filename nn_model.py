import numpy as np

class SimpleNN:
    def __init__(self, input_size=4, hidden_size=8, output_size=3):
        # Random initialization
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)
   
    def relu_derivative(self, x):
        return (x > 0).astype(float)
   
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
   
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
   
    def compute_loss(self, output, y):
        # Cross-entropy loss
        m = y.shape[0]
        log_probs = -np.log(output[np.arange(m), y] + 1e-8)
        return np.mean(log_probs)
   
    def backward(self, X, y, learning_rate):
        m = y.shape[0]
       
        # Output layer gradient
        dz2 = self.a2.copy()
        dz2[np.arange(m), y] -= 1
        dz2 /= m
       
        # Backprop
        self.W2 -= learning_rate * np.dot(self.a1.T, dz2)
        self.b2 -= learning_rate * np.sum(dz2, axis=0, keepdims=True)
       
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
       
        self.W1 -= learning_rate * np.dot(X.T, dz1)
        self.b1 -= learning_rate * np.sum(dz1, axis=0, keepdims=True)
   
    def train_epoch(self, X, y, learning_rate):
        output = self.forward(X)
        loss = self.compute_loss(output, y)
        self.backward(X, y, learning_rate)
       
        # Accuracy
        predictions = np.argmax(output, axis=1)
        accuracy = np.mean(predictions == y)
        return loss, accuracy


