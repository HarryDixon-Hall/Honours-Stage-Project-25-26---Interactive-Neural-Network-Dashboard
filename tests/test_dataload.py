import numpy as np
import pytest

from dataload import (
    load_dataset_iris,
    load_dataset_digits,
    load_dataset_wine,
    load_dataset,
    get_dataset_stats,
)

#to assert in python unit tests is to test the ASSUMPTIONS
#this should check a condition for an expected value
#an unexptected value should raise a false bool value?

def assert_common_dataset(X_train, X_test, y_train, y_test, meta, expected_name):
    #the shapes of values in dataset are not empty
    assert X_train.ndim == 2 #number of array dimensions, this is 2
    assert X_test.ndim == 2
    assert y_train.ndim == 1
    assert y_test.ndim == 1
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    #Standardisation - zero mean,
    X_full = np.vstack([X_train, X_test]) #the arrays are stacked veritcally (as rows here)
    means = X_full.mean(axis=0)
    stds = X_full.std(axis=0) #the standard deviation is zero atm
    assert np.allclose(means, 0, atol=1e-1)
    assert np.allclose(stds, 1, atol=2e-1) #absolute tolerance parameter - array_like

    #meta information
    assert meta["name"] == expected_name
    assert meta["n_features"] == X_train.shape[1]
    assert meta["n_classes"] == len(np.unique(np.concatenate([y_train, y_test])))
    assert len(meta["feature_names"]) == meta["n_features"]
    assert len(meta["class_names"]) == meta["n_classes"]

    def test_load_dataset_iris():
        X_train, X_test, y_train, y_test, meta = load_dataset_iris()
        assert_common_dataset(X_train, X_test, y_train, y_test, meta, "Iris")


    def test_load_dataset_wine():
        X_train, X_test, y_train, y_test, meta = load_dataset_wine()
        assert_common_dataset(X_train, X_test, y_train, y_test, meta, "Wine")


    def test_load_dataset_digits():
        X_train, X_test, y_train, y_test, meta = load_dataset_digits()
        assert_common_dataset(X_train, X_test, y_train, y_test, meta, "Digits")

    
    def test_valid_datasetnames():
        for name in ["iris", "wine", "digits"]:
            X_train, X_test, y_train, y_test, meta = load_dataset(name) # this will check if the common load dataset is retrieving a valid name
            assert_common_dataset(X_train, X_test, y_train, y_test, meta, meta["name"])

    def test_invalid_datasetnames():
        with pytest.raises(ValueError):
            load_dataset("unknown_dataset") # if the the value is wrong 

    #need to add a test for dataset stats?
    #how to do it since each dataset will have different features, classes etc

