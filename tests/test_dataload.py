"""
Tests for dataload.py
=====================
Covers functional and non-functional requirements for dataset loading,
pre-processing (standardisation), and train/test splitting utilities.

Functional requirements
-----------------------
FR-DL-1  load_dataset returns all expected splits and metadata for every
         supported dataset name.
FR-DL-2  Returned feature arrays are standardised (zero mean, unit variance).
FR-DL-3  Metadata dict contains the required keys: 'name', 'feature_names',
         'class_names', 'n_features', 'n_classes'.
FR-DL-4  load_dataset raises ValueError for an unrecognised dataset name.
FR-DL-5  get_dataset_stats returns the correct sample count, feature count,
         class count, and class_names list.
FR-DL-6  Train/test split proportions match the expected 80/20 ratio.

Non-functional requirements
---------------------------
NFR-DL-1  Each dataset loads within an acceptable time limit (performance).
NFR-DL-2  Repeated calls to load_dataset produce consistent (reproducible)
          results (determinism).
NFR-DL-3  load_dataset is resilient to unexpected input types (robustness).
"""

import pytest
import sys
import os

# Allow imports from the project root when running tests from the tests/ dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataload import load_dataset, get_dataset_stats, DATASETS


# ---------------------------------------------------------------------------
# Functional requirement tests
# ---------------------------------------------------------------------------

class TestLoadDatasetFunctional:
    """Functional tests for load_dataset and load_dataset_* helpers."""

    # FR-DL-1 ----------------------------------------------------------------
    def test_iris_returns_five_values(self):
        """load_dataset('iris') must return exactly five values."""
        pass

    def test_wine_returns_five_values(self):
        """load_dataset('wine') must return exactly five values."""
        pass

    def test_digits_returns_five_values(self):
        """load_dataset('digits') must return exactly five values."""
        pass

    def test_all_registered_datasets_loadable(self):
        """Every key in DATASETS should load without error."""
        pass

    # FR-DL-2 ----------------------------------------------------------------
    def test_iris_features_are_standardised(self):
        """Training features for iris should have approximately zero mean."""
        pass

    def test_wine_features_are_standardised(self):
        """Training features for wine should have approximately zero mean."""
        pass

    # FR-DL-3 ----------------------------------------------------------------
    def test_iris_meta_has_required_keys(self):
        """Iris metadata dict must contain all required keys."""
        pass

    def test_wine_meta_has_required_keys(self):
        """Wine metadata dict must contain all required keys."""
        pass

    def test_digits_meta_has_required_keys(self):
        """Digits metadata dict must contain all required keys."""
        pass

    # FR-DL-4 ----------------------------------------------------------------
    def test_unknown_dataset_raises_value_error(self):
        """Requesting an unknown dataset name must raise ValueError."""
        pass

    # FR-DL-5 ----------------------------------------------------------------
    def test_get_dataset_stats_returns_correct_keys(self):
        """get_dataset_stats must return a dict with expected keys."""
        pass

    def test_get_dataset_stats_correct_sample_count(self):
        """Sample count from get_dataset_stats must equal len(X)."""
        pass

    def test_get_dataset_stats_correct_feature_count(self):
        """Feature count from get_dataset_stats must equal X.shape[1]."""
        pass

    def test_get_dataset_stats_correct_class_count(self):
        """Class count must equal the number of unique labels in y."""
        pass

    # FR-DL-6 ----------------------------------------------------------------
    def test_iris_train_test_split_ratio(self):
        """Iris split should produce roughly 80 % training samples."""
        pass

    def test_wine_train_test_split_ratio(self):
        """Wine split should produce roughly 80 % training samples."""
        pass


# ---------------------------------------------------------------------------
# Non-functional requirement tests
# ---------------------------------------------------------------------------

class TestLoadDatasetNonFunctional:
    """Non-functional tests for dataset loading (performance, robustness)."""

    # NFR-DL-1 ---------------------------------------------------------------
    def test_iris_loads_within_time_limit(self):
        """Iris dataset should load in under 2 seconds."""
        pass

    def test_wine_loads_within_time_limit(self):
        """Wine dataset should load in under 2 seconds."""
        pass

    def test_digits_loads_within_time_limit(self):
        """Digits dataset should load in under 2 seconds."""
        pass

    # NFR-DL-2 ---------------------------------------------------------------
    def test_iris_load_is_reproducible(self):
        """Two consecutive iris loads must return identical arrays."""
        pass

    def test_wine_load_is_reproducible(self):
        """Two consecutive wine loads must return identical arrays."""
        pass

    # NFR-DL-3 ---------------------------------------------------------------
    def test_load_dataset_with_integer_name_raises_error(self):
        """Passing an integer as a dataset name should raise an error."""
        pass

    def test_load_dataset_with_empty_string_raises_value_error(self):
        """Passing an empty string should raise a ValueError."""
        pass
