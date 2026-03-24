#DECLARATION: CODE HERE IS ASSISTED BY https://www.perplexity.ai/ 04/12/25 - 22/03/26
"""
conftest.py — Shared pytest fixtures for the Interactive Neural Network Dashboard test suite.

Fixtures defined here are automatically available to all test modules without explicit imports.
They provide reusable, pre-built objects (datasets, model instances, etc.) so individual tests
stay focused on behaviour rather than setup boilerplate.
"""

import numpy as np
import pytest

import sys
import os

# Ensure the project root is on the path so all modules are importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_X_train():
    """Tiny synthetic feature matrix (20 samples × 4 features) for fast unit tests."""
    np.random.seed(0)
    return np.random.randn(20, 4).astype(np.float64)


@pytest.fixture
def small_y_train():
    """Corresponding integer class labels (3 classes) for *small_X_train*."""
    np.random.seed(0)
    return np.random.randint(0, 3, size=20)


@pytest.fixture
def small_X_test():
    """Tiny synthetic test feature matrix (10 samples × 4 features)."""
    np.random.seed(1)
    return np.random.randn(10, 4).astype(np.float64)


@pytest.fixture
def small_y_test():
    """Corresponding integer class labels for *small_X_test*."""
    np.random.seed(1)
    return np.random.randint(0, 3, size=10)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def logistic_regression_model():
    """A fresh LogisticRegression instance with reproducible weights."""
    from models import LogisticRegression
    return LogisticRegression(input_size=4, output_size=3, seed=42)


@pytest.fixture
def simple_nn_model():
    """A fresh SimpleNN instance (1 hidden layer) with reproducible weights."""
    from models import SimpleNN
    return SimpleNN(input_size=4, hidden_size=8, output_size=3, seed=42)


@pytest.fixture
def complex_nn_model():
    """A fresh ComplexNN instance (2 hidden layers) with reproducible weights."""
    from models import ComplexNN
    return ComplexNN(input_size=4, hidden1_size=8, hidden2_size=8, output_size=3, seed=42)


# ---------------------------------------------------------------------------
# Trainer fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def built_logistic_regression():
    """LogisticRegression model produced via the trainer.build_model factory."""
    from trainer import build_model
    return build_model("Logistic Regression", input_size=4, output_size=3, seed=42)


@pytest.fixture
def built_simple_nn():
    """SimpleNN model produced via the trainer.build_model factory."""
    from trainer import build_model
    return build_model("simple_nn", input_size=4, output_size=3, hidden_size=8, seed=42)


# ---------------------------------------------------------------------------
# Dataset loading fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def iris_data():
    """Pre-loaded Iris dataset splits and metadata (session-scoped for speed)."""
    from dataload import load_dataset
    return load_dataset("iris")


@pytest.fixture(scope="session")
def wine_data():
    """Pre-loaded Wine dataset splits and metadata (session-scoped for speed)."""
    from dataload import load_dataset
    return load_dataset("wine")
