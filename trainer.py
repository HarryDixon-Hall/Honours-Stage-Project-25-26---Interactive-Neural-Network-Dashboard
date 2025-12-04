from nn_model import SimpleNN
import numpy as np

def train_model(X_train, y_train, epochs=50, learning_rate=0.01, hidden_size=8):
    """Train and return model + history"""
    model = SimpleNN(input_size=X_train.shape[1],
                     hidden_size=hidden_size,
                     output_size=len(np.unique(y_train)))
   
    history = {'loss': [], 'accuracy': []}
   
    for epoch in range(epochs):
        loss, acc = model.train_epoch(X_train, y_train, learning_rate)
        history['loss'].append(loss)
        history['accuracy'].append(acc)
   
    return model, history
 