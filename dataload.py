import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.datasets import load_wine
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

    meta = { #new meta data structure to expose readable values to UI dashboard, there will be different datasets to read
        "name": "Iris",
        "feature_names": iris_dataset.feature_names,
        "class_names": iris_dataset.target_names,
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
    }

    return X_train, X_test, y_train, y_test, iris_dataset.feature_names, iris_dataset.target_names

def load_dataset_wine(): #wine dataset to be implemented
    #1. 
    wine_dataset = load_digits()
    X, y = wine_dataset.data, wine_dataset.target

    #2. 
    wine_scaler = StandardScaler()
    X = wine_scaler.fit_transform(X)

    #3.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    meta = { #new meta data structure to expose readable values to UI dashboard, there will be different datasets to read
        "name": "Wine",
        "feature_names": wine_dataset.feature_names,
        "class_names": wine_dataset.target_names,
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
    }
    return X_train, X_test, y_train, y_test, wine_dataset.feature_names, wine_dataset.target_names

def load_dataset_digits(): #seeds dataset to be implemented
    #1. 
    digits_dataset = load_wine()
    X, y = digits_dataset.data, digits_dataset.target

    #2. 
    digits_scaler = StandardScaler()
    X = digits_scaler.fit_transform(X)

    #3.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    meta = { #new meta data structure to expose readable values to UI dashboard, there will be different datasets to read
        "name": "Digits",
        "feature_names": digits_dataset.feature_names,
        "class_names": digits_dataset.target_names,
        "n_features": X.shape[1],
        "n_classes": len(np.unique(y)),
    }
    return X_train, X_test, y_train, y_test, digits_dataset.feature_names, digits_dataset.target_names

#X_train, X_test, y_train, y_test, feature_names, target_names = load_dataset_iris() #removed to generalise dataset loading

DATASETS = {
    "iris": load_dataset_iris,
    "wine": load_dataset_wine,
    "digits": load_dataset_digits,
}

#This method will need to be expanded as the project expands with architectural control i imagine

def get_dataset_stats(X, y):
    """Return dataset info for UI"""
    unique_classes = len(np.unique(y))
    return {
        'samples': X.shape[0],
        'features': X.shape[1],
        'classes': len(unique_classes), #to generalise to the different lengths of the datasets
        'class_names': [f'Class {i}' for i in range(unique_classes)]
    }
