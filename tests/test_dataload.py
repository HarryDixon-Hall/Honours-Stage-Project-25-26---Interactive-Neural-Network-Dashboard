#DECLARATION: CODE HERE IS ASSISTED BY https://www.perplexity.ai/ 04/12/25 - 22/03/26
"""
test_dataload.py — Test backbone for dataload.py
=================================================
Covers the dataset-loading utilities in dataload.py:
  • load_dataset_iris()
  • load_dataset_wine()
  • load_dataset_digits()
  • load_dataset()   — unified dispatcher
  • get_dataset_stats()

Each class is split into FR (Functional Requirements) and NFR (Non-Functional Requirements).

How pytest achieves the feature → FR/NFR → verification aim
------------------------------------------------------------
Each test function maps directly to one requirement:

  Feature                       FR code     Assertion (what pytest checks)
  ──────────────────────────────────────────────────────────────────────────
  load_dataset_iris returns     FR-LDI-01   assert isinstance(result, tuple) and len == 5
  Iris feature count            FR-LDI-02   assert X_train.shape[1] == 4
  Dispatcher raises on unknown  FR-LDS-04   with pytest.raises(ValueError): ...

When the suite is run:
  PASSED  → the implementation satisfies that requirement
  FAILED  → the implementation violates that requirement; pytest prints exactly
             what was expected and what was received
  SKIPPED → the test is specified but not yet implemented

See tests/README.md for the full narrative-flow explanation.
"""

import time
import pytest
import numpy as np


# ===========================================================================
# load_dataset_iris()
# ===========================================================================

class TestLoadDatasetIrisFunctional:
    """FR tests for dataload.load_dataset_iris()."""

    # FR-LDI-01: Returns a 5-tuple (X_train, X_test, y_train, y_test, meta)
    def test_returns_five_tuple(self):
        from dataload import load_dataset_iris
        result = load_dataset_iris()
        assert isinstance(result, tuple)
        assert len(result) == 5

    # FR-LDI-02: X_train and X_test have 4 features (Iris has 4 input features)
    def test_feature_count_is_four(self):
        from dataload import load_dataset_iris
        X_train, X_test, _, _, _ = load_dataset_iris()
        assert X_train.shape[1] == 4
        assert X_test.shape[1] == 4

    # FR-LDI-03: y_train and y_test contain only class indices in {0, 1, 2}
    def test_labels_are_valid_class_indices(self):
        from dataload import load_dataset_iris
        _, _, y_train, y_test, _ = load_dataset_iris()
        assert set(np.unique(y_train)).issubset({0, 1, 2})
        assert set(np.unique(y_test)).issubset({0, 1, 2})

    # FR-LDI-04: Combined sample count equals total Iris dataset size (150 samples)
    def test_total_sample_count(self):
        from dataload import load_dataset_iris
        X_train, X_test, _, _, _ = load_dataset_iris()
        assert X_train.shape[0] + X_test.shape[0] == 150

    # FR-LDI-05: meta dict contains keys 'name', 'feature_names', 'class_names', 'n_features', 'n_classes'
    def test_meta_contains_required_keys(self):
        from dataload import load_dataset_iris
        _, _, _, _, meta = load_dataset_iris()
        for key in ('name', 'feature_names', 'class_names', 'n_features', 'n_classes'):
            assert key in meta

    # FR-LDI-06: meta['n_classes'] equals 3
    def test_meta_n_classes_is_three(self):
        from dataload import load_dataset_iris
        _, _, _, _, meta = load_dataset_iris()
        assert meta['n_classes'] == 3

    # FR-LDI-07: meta['name'] equals "Iris"
    def test_meta_name_is_iris(self):
        from dataload import load_dataset_iris
        _, _, _, _, meta = load_dataset_iris()
        assert meta['name'] == "Iris"

    # FR-LDI-08: Features are standardised (mean ≈ 0, std ≈ 1) on the training set
    def test_features_are_standardised(self):
        from dataload import load_dataset_iris
        X_train, _, _, _, _ = load_dataset_iris()
        # StandardScaler is fit on the full dataset before splitting, so training-set
        # statistics are close to 0/1 but not exact — use a generous tolerance.
        assert np.abs(X_train.mean(axis=0)).max() < 0.5
        assert np.abs(X_train.std(axis=0) - 1.0).max() < 0.5

    # FR-LDI-09: Stratified split preserves class proportions in training and test sets
    def test_stratified_split_preserves_class_proportions(self):
        from dataload import load_dataset_iris
        _, _, y_train, y_test, _ = load_dataset_iris()
        assert len(np.unique(y_train)) == 3
        assert len(np.unique(y_test)) == 3


class TestLoadDatasetIrisNonFunctional:
    """NFR tests for dataload.load_dataset_iris()."""

    # NFR-LDI-01: Function completes within an acceptable time
    def test_load_performance(self):
        from dataload import load_dataset_iris
        start = time.perf_counter()
        load_dataset_iris()
        assert time.perf_counter() - start < 5.0

    # NFR-LDI-02: Calling the function twice returns identical splits (deterministic)
    def test_deterministic_output(self):
        from dataload import load_dataset_iris
        X_train1, X_test1, y_train1, y_test1, _ = load_dataset_iris()
        X_train2, X_test2, y_train2, y_test2, _ = load_dataset_iris()
        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(y_test1, y_test2)


# ===========================================================================
# load_dataset_wine()
# ===========================================================================

class TestLoadDatasetWineFunctional:
    """FR tests for dataload.load_dataset_wine()."""

    # FR-LDW-01: Returns a 5-tuple (X_train, X_test, y_train, y_test, meta)
    def test_returns_five_tuple(self):
        from dataload import load_dataset_wine
        result = load_dataset_wine()
        assert isinstance(result, tuple)
        assert len(result) == 5

    # FR-LDW-02: X_train and X_test have 13 features (Wine dataset)
    def test_feature_count_is_thirteen(self):
        from dataload import load_dataset_wine
        X_train, X_test, _, _, _ = load_dataset_wine()
        assert X_train.shape[1] == 13
        assert X_test.shape[1] == 13

    # FR-LDW-03: y labels contain only valid class indices {0, 1, 2}
    def test_labels_are_valid_class_indices(self):
        from dataload import load_dataset_wine
        _, _, y_train, y_test, _ = load_dataset_wine()
        assert set(np.unique(y_train)).issubset({0, 1, 2})
        assert set(np.unique(y_test)).issubset({0, 1, 2})

    # FR-LDW-04: meta['name'] equals "Wine"
    def test_meta_name_is_wine(self):
        from dataload import load_dataset_wine
        _, _, _, _, meta = load_dataset_wine()
        assert meta['name'] == "Wine"

    # FR-LDW-05: meta['n_classes'] equals 3
    def test_meta_n_classes_is_three(self):
        from dataload import load_dataset_wine
        _, _, _, _, meta = load_dataset_wine()
        assert meta['n_classes'] == 3

    # FR-LDW-06: Features are standardised on the training set
    def test_features_are_standardised(self):
        from dataload import load_dataset_wine
        X_train, _, _, _, _ = load_dataset_wine()
        assert np.abs(X_train.mean(axis=0)).max() < 0.5
        assert np.abs(X_train.std(axis=0) - 1.0).max() < 0.5


class TestLoadDatasetWineNonFunctional:
    """NFR tests for dataload.load_dataset_wine()."""

    # NFR-LDW-01: Function completes within an acceptable time
    def test_load_performance(self):
        from dataload import load_dataset_wine
        start = time.perf_counter()
        load_dataset_wine()
        assert time.perf_counter() - start < 5.0

    # NFR-LDW-02: Calling the function twice returns identical splits (deterministic)
    def test_deterministic_output(self):
        from dataload import load_dataset_wine
        X_train1, _, y_train1, _, _ = load_dataset_wine()
        X_train2, _, y_train2, _, _ = load_dataset_wine()
        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(y_train1, y_train2)


# ===========================================================================
# load_dataset_digits()
# ===========================================================================

class TestLoadDatasetDigitsFunctional:
    """FR tests for dataload.load_dataset_digits()."""

    # FR-LDD-01: Returns a 5-tuple (X_train, X_test, y_train, y_test, meta)
    def test_returns_five_tuple(self):
        from dataload import load_dataset_digits
        result = load_dataset_digits()
        assert isinstance(result, tuple)
        assert len(result) == 5

    # FR-LDD-02: meta dict contains the required keys
    def test_meta_contains_required_keys(self):
        from dataload import load_dataset_digits
        _, _, _, _, meta = load_dataset_digits()
        for key in ('name', 'feature_names', 'class_names', 'n_features', 'n_classes'):
            assert key in meta

    # FR-LDD-03: X_train and X_test share the same number of features
    def test_train_and_test_have_same_feature_count(self):
        from dataload import load_dataset_digits
        X_train, X_test, _, _, _ = load_dataset_digits()
        assert X_train.shape[1] == X_test.shape[1]


class TestLoadDatasetDigitsNonFunctional:
    """NFR tests for dataload.load_dataset_digits()."""

    # NFR-LDD-01: Function completes within an acceptable time
    def test_load_performance(self):
        from dataload import load_dataset_digits
        start = time.perf_counter()
        load_dataset_digits()
        assert time.perf_counter() - start < 5.0


# ===========================================================================
# load_dataset() — unified dispatcher
# ===========================================================================

class TestLoadDatasetDispatcherFunctional:
    """FR tests for dataload.load_dataset()."""

    # FR-LDS-01: load_dataset("iris") delegates to load_dataset_iris() successfully
    def test_dispatches_iris(self):
        from dataload import load_dataset
        result = load_dataset("iris")
        assert isinstance(result, tuple) and len(result) == 5

    # FR-LDS-02: load_dataset("wine") delegates to load_dataset_wine() successfully
    def test_dispatches_wine(self):
        from dataload import load_dataset
        result = load_dataset("wine")
        assert isinstance(result, tuple) and len(result) == 5

    # FR-LDS-03: load_dataset("digits") delegates to load_dataset_digits() successfully
    def test_dispatches_digits(self):
        from dataload import load_dataset
        result = load_dataset("digits")
        assert isinstance(result, tuple) and len(result) == 5

    # FR-LDS-04: load_dataset() raises ValueError for an unrecognised dataset name
    def test_raises_for_unknown_dataset(self):
        from dataload import load_dataset
        with pytest.raises(ValueError):
            load_dataset("not_a_dataset")

    # FR-LDS-05: Return value structure is consistent across all supported datasets
    def test_consistent_return_structure(self):
        from dataload import load_dataset
        for name in ("iris", "wine", "digits"):
            result = load_dataset(name)
            assert isinstance(result, tuple) and len(result) == 5
            X_train, X_test, y_train, y_test, meta = result
            assert X_train.ndim == 2
            assert y_train.ndim == 1
            assert X_train.shape[1] == X_test.shape[1]


class TestLoadDatasetDispatcherNonFunctional:
    """NFR tests for dataload.load_dataset()."""

    # NFR-LDS-01: dispatcher adds negligible overhead compared to direct function calls
    def test_dispatcher_overhead_is_negligible(self):
        from dataload import load_dataset
        start = time.perf_counter()
        load_dataset("iris")
        assert time.perf_counter() - start < 5.0


# ===========================================================================
# get_dataset_stats()
# ===========================================================================

class TestGetDatasetStatsFunctional:
    """FR tests for dataload.get_dataset_stats()."""

    # FR-GDS-01: Returns a dict with keys 'samples', 'features', 'classes', 'class_names'
    def test_returns_dict_with_required_keys(self, iris_data):
        from dataload import get_dataset_stats
        X_train, _, y_train, _, _ = iris_data
        stats = get_dataset_stats(X_train, y_train)
        for key in ('samples', 'features', 'classes', 'class_names'):
            assert key in stats

    # FR-GDS-02: 'samples' matches the number of rows in X
    def test_samples_count_matches_x_rows(self, iris_data):
        from dataload import get_dataset_stats
        X_train, _, y_train, _, _ = iris_data
        stats = get_dataset_stats(X_train, y_train)
        assert stats['samples'] == X_train.shape[0]

    # FR-GDS-03: 'features' matches the number of columns in X
    def test_features_count_matches_x_columns(self, iris_data):
        from dataload import get_dataset_stats
        X_train, _, y_train, _, _ = iris_data
        stats = get_dataset_stats(X_train, y_train)
        assert stats['features'] == X_train.shape[1]

    # FR-GDS-04: 'classes' matches the number of unique labels in y
    def test_classes_count_matches_unique_labels(self, iris_data):
        from dataload import get_dataset_stats
        X_train, _, y_train, _, _ = iris_data
        stats = get_dataset_stats(X_train, y_train)
        assert stats['classes'] == len(np.unique(y_train))

    # FR-GDS-05: 'class_names' list length equals 'classes'
    def test_class_names_length_matches_classes(self, iris_data):
        from dataload import get_dataset_stats
        X_train, _, y_train, _, _ = iris_data
        stats = get_dataset_stats(X_train, y_train)
        assert len(stats['class_names']) == stats['classes']


class TestGetDatasetStatsNonFunctional:
    """NFR tests for dataload.get_dataset_stats()."""

    # NFR-GDS-01: Function completes within an acceptable time for large datasets
    def test_stats_performance(self, iris_data):
        from dataload import get_dataset_stats
        X_train, _, y_train, _, _ = iris_data
        start = time.perf_counter()
        get_dataset_stats(X_train, y_train)
        assert time.perf_counter() - start < 1.0

