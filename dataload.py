import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

'''
iris = load_iris()

plt.scatter(iris.data[:,0], iris.data[:, 1], c=iris.target)
plt.xlabel('Sepal length(cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Sepal Length vs Sepal Width')
plt.show()
'''

#Methods to load the dataset and aquire its features

#Sequence
#1. Load iris datset
#2. Standardise it with StandardScaler 
#3. Split into train/test  

def load_dataset_iris():

    #1. 
    iris_dataset = load_iris()
    X, y = iris_dataset.data, iris_dataset.target

    #2. 
    iris_scaler = StandardScaler()
    X = iris_scaler.fit_transform(X)

    #3.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, iris_dataset.feature_names, iris_dataset.target_names

X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset_iris()

