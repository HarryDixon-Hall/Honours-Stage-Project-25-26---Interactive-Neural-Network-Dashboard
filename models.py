import numpy as np

class LogisticRegression: #same as SimpleNN but without hidden layer
    def __init__(self, input_size, output_size, seed=None):
        np.random.seed(seed)
        self.W = 0.01 * np.random.randn(input_size, output_size)
        self.b = np.zeros((1, output_size))

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, X):
        logits = X @ self.W + self.b
        self.probs = self._softmax(logits)
        return self.probs

    def compute_loss(self, output, y):
        m = y.shape[0]
        log_probs = -np.log(output[np.arange(m), y] + 1e-8)
        return np.mean(log_probs)

    def backward(self, X, y, learning_rate):
        m = y.shape[0]
        grad = self.probs.copy()
        grad[np.arange(m), y] -= 1
        grad /= m

        dW = X.T @ grad
        db = np.sum(grad, axis=0, keepdims=True)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

    def train_epoch(self, X, y, learning_rate):
        output = self.forward(X)
        loss = self.compute_loss(output, y)
        self.backward(X, y, learning_rate)

        preds = np.argmax(output, axis=1)
        acc = np.mean(preds == y)
        return loss, acc

class SimpleNN: #1 hidden layer (shallow NN)
    def __init__(self, input_size=4, hidden_size=8, output_size=3, seed = None):
        #Reproducible weight initalisation
        np.random.seed(seed)

        # Random initialisation
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

class ComplexNN: #2 hidden layers (deep NN)
    def __init__(self, input_size=4, hidden1_size=8, hidden2_size = 8, output_size=3, seed = None):
        #Reproducible weight initalisation
        np.random.seed(seed)

        # Random initialisation

        #layer 1: input to hidden layer 1
        self.W1 = np.random.randn(input_size, hidden1_size) * 0.01
        self.b1 = np.zeros((1, hidden1_size))

        #layer 2: hidden layer 1 to hidden layer 2
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * 0.01
        self.b2 = np.zeros((1, hidden2_size))

        #layer 3: hidden layer 2 to output
        self.W3 = np.random.randn(hidden2_size, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))

    #activation functions
    def relu(self, x):
        return np.maximum(0, x) #(0, x), the max for activation
   
    def relu_derivative(self, x):
        return (x > 0).astype(float) #Relu derivative, x>0, or else it will be 0
   
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
   
   #forward pass of data through hidden layers
    def forward(self, X):
        #hidden layer 1
        self.z1 = np.dot(X, self.W1) + self.b1       #pre-activation
        self.a1 = self.relu(self.z1)                 #activation

        #hidden layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2 #pre-activation
        self.a2 = self.relu(self.z2)                 #activation

        #output layer 
        self.z3 = np.dot(self.a2, self.W3) + self.b3 #pre-activation
        self.a3 = self.softmax(self.z3)              #activation
        
        return self.a3 #return total output probabilities
   
   #data loss in prediction error
    def compute_loss(self, output, y):
        # Cross-entropy loss
        m = y.shape[0]
        log_probs = -np.log(output[np.arange(m), y] + 1e-8)
        return np.mean(log_probs)
   
   #backward pass of data
    def backward(self, X, y, learning_rate):
        m = y.shape[0]

        dz3 = self.a3.copy()       #start from the output probabilities given by forward pass
        dz3[np.arrange(m), y] -= 1 #subtract 1 at the true class index  
        dz3 /= m                   #new average over batch


        #backprop path: output => HL2 => HL1

        #output layer gradients
        dW3 = np.dot(self.a2.T, X) 
        db3 = np.sum(dz3, axis=0, keepdims=True)

        #backprop for hidden layer 2
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz1, axis = 0, keepdim=True)

        #backprop for hidden layer 1
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(self.a1.T, dz1)
        db1 = np.sum(dz1, axis = 0, keepdim=True)

        #gradient descent updates after backpropagation
        self.W3 -= learning_rate * dW3 #output weights
        self.b3 -= learning_rate * dW3 #output biases
        self.W2 -= learning_rate * dW2 #hidden layer 2 weights
        self.b2 -= learning_rate * db2 #hidden layer 2 biases
        self.W1 -= learning_rate * dW1 #hidden layer 1 weights
        self.b1 -= learning_rate * dW1 #hidden layer 1 weights
       

    #method for training a single epoch   
    def train_epoch(self, X, y, learning_rate):
        output = self.forward(X) #forward pass
        loss = self.compute_loss(output, y) #cross-entrpy loss
        self.backward(X, y, learning_rate)  #backprop then updates weights and biases
       
        # Accuracy
        predictions = np.argmax(output, axis=1) 
        accuracy = np.mean(predictions == y)
        return loss, accuracy




