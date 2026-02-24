from models import LogisticRegression
from models import SimpleNN
from models import ComplexNN

import numpy as np

def build_model(model_name, input_size, output_size, hidden_size=8, seed=42): #build a model using the parameter values onto its layer map
    np.random.seed(seed)

    if model_name == "Logistic Regression": #Logistic Regression
        return LogisticRegression(
            input_size=input_size,
            output_size=output_size, 
            seed=seed,
        )
    
    if model_name == "simple_nn": #NN-1-Layer
        return SimpleNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size, 
            seed=seed,
        )
    
    if model_name == "NN-2-Layer": #NN-2-Layer
        return ComplexNN(
            input_size=input_size,
            hidden1_size=hidden_size,
            hidden2_size=hidden_size,
            output_size=output_size, 
            seed=seed,
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}") #the value error should now be interpolated
    


def train_model(model, X_train, y_train, epochs=50, learning_rate=0.01,): #build functionality pulled out of this method to "build_model"
    
    history = {'loss': [], 'accuracy': []}
   
    for epoch in range(epochs):
        loss, acc = model.train_epoch(X_train, y_train, learning_rate)
        history['loss'].append(loss)
        history['accuracy'].append(acc)
   
    return model, history
 