import numpy as np

from modelFactory.dataload import get_dataset_stats, load_dataset


def test_get_dataset_stats_returns_expected_counts():
    features = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    labels = np.array([0, 1, 0, 1])

    stats = get_dataset_stats(features, labels)

    assert stats == {
        "samples": 4,
        "features": 2,
        "classes": 2,
        "class_names": ["Class 0", "Class 1"],
    }


def test_load_dataset_iris_returns_consistent_shapes():
    x_train, x_test, y_train, y_test, meta = load_dataset("iris")

    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    assert x_train.shape[1] == meta["n_features"]
    assert meta["name"] == "Iris"
    assert meta["n_classes"] == 3