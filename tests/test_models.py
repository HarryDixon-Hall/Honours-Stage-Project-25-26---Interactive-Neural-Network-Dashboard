#DECLARATION: CODE HERE IS ASSISTED BY https://www.perplexity.ai/ 04/12/25 - 22/03/26
"""
test_models.py — Test backbone for models.py
=============================================
Covers the three neural-network classes defined in models.py:
  • LogisticRegression  (no hidden layer)
  • SimpleNN            (1 hidden layer)
  • ComplexNN           (2 hidden layers)

Each test class is split into two sections:
  FR  — Functional Requirements  : what the component *does* (correct outputs, shapes, behaviour)
  NFR — Non-Functional Requirements : quality attributes (speed, reproducibility, numerical stability)

All test bodies call ``pytest.skip("Not yet implemented")`` so that pytest reports them as
SKIPPED rather than falsely PASSED.  Replace the skip call with real assertions to implement
each test.  See tests/README.md for an explanation of why pytest.skip() is used here and
for the difference between pytest and Python's built-in unittest module.
"""

import pytest
import numpy as np


# ===========================================================================
# LogisticRegression
# ===========================================================================

class TestLogisticRegressionFunctional:
    """FR tests for models.LogisticRegression."""

    # FR-LR-01: Model initialises and weight / bias tensors have the correct shapes
    def test_weight_shapes_on_init(self, logistic_regression_model):
        pytest.skip("Not yet implemented")

    # FR-LR-02: forward() returns a probability distribution (all values in [0, 1], rows sum to 1)
    def test_forward_returns_valid_probabilities(self, logistic_regression_model, small_X_train):
        pytest.skip("Not yet implemented")

    # FR-LR-03: forward() output shape is (n_samples, n_classes)
    def test_forward_output_shape(self, logistic_regression_model, small_X_train):
        pytest.skip("Not yet implemented")

    # FR-LR-04: compute_loss() returns a non-negative scalar
    def test_compute_loss_is_non_negative(self, logistic_regression_model, small_X_train, small_y_train):
        pytest.skip("Not yet implemented")

    # FR-LR-05: backward() updates weights (W and b change after a single backward call)
    def test_backward_updates_weights(self, logistic_regression_model, small_X_train, small_y_train):
        pytest.skip("Not yet implemented")

    # FR-LR-06: train_epoch() returns (loss, accuracy) as a 2-tuple
    def test_train_epoch_returns_loss_and_accuracy(self, logistic_regression_model, small_X_train, small_y_train):
        pytest.skip("Not yet implemented")

    # FR-LR-07: Accuracy returned by train_epoch() is bounded in [0, 1]
    def test_train_epoch_accuracy_bounded(self, logistic_regression_model, small_X_train, small_y_train):
        pytest.skip("Not yet implemented")

    # FR-LR-08: Loss decreases over multiple training epochs on a learnable problem
    def test_loss_decreases_over_epochs(self, small_X_train, small_y_train):
        pytest.skip("Not yet implemented")


class TestLogisticRegressionNonFunctional:
    """NFR tests for models.LogisticRegression."""

    # NFR-LR-01: forward() completes within an acceptable time budget for typical input sizes
    def test_forward_performance(self, logistic_regression_model, small_X_train):
        pytest.skip("Not yet implemented")

    # NFR-LR-02: Using the same seed produces identical weights on separate instances
    def test_reproducibility_with_seed(self):
        pytest.skip("Not yet implemented")

    # NFR-LR-03: Model handles edge-case input (single sample) without raising an exception
    def test_single_sample_input(self, logistic_regression_model):
        pytest.skip("Not yet implemented")

    # NFR-LR-04: Model handles large input batches without memory errors
    def test_large_batch_input(self, logistic_regression_model):
        pytest.skip("Not yet implemented")


# ===========================================================================
# SimpleNN (1 hidden layer)
# ===========================================================================

class TestSimpleNNFunctional:
    """FR tests for models.SimpleNN."""

    # FR-SNN-01: Model initialises with correct shapes for W1, b1, W2, b2
    def test_weight_shapes_on_init(self, simple_nn_model):
        pytest.skip("Not yet implemented")

    # FR-SNN-02: relu() returns zero for negative inputs and is identity for positive inputs
    def test_relu_activation(self, simple_nn_model):
        pytest.skip("Not yet implemented")

    # FR-SNN-03: relu_derivative() returns 0 for x <= 0 and 1 for x > 0
    def test_relu_derivative(self, simple_nn_model):
        pytest.skip("Not yet implemented")

    # FR-SNN-04: softmax() output rows sum to 1 and all values are in (0, 1)
    def test_softmax_output(self, simple_nn_model):
        pytest.skip("Not yet implemented")

    # FR-SNN-05: forward() stores intermediate activations (a1, z1, a2, z2) as instance attributes
    def test_forward_stores_intermediate_activations(self, simple_nn_model, small_X_train):
        pytest.skip("Not yet implemented")

    # FR-SNN-06: forward() output shape is (n_samples, n_classes)
    def test_forward_output_shape(self, simple_nn_model, small_X_train):
        pytest.skip("Not yet implemented")

    # FR-SNN-07: compute_loss() returns a non-negative scalar
    def test_compute_loss_is_non_negative(self, simple_nn_model, small_X_train, small_y_train):
        pytest.skip("Not yet implemented")

    # FR-SNN-08: backward() updates all four parameter tensors (W1, b1, W2, b2)
    def test_backward_updates_all_weights(self, simple_nn_model, small_X_train, small_y_train):
        pytest.skip("Not yet implemented")

    # FR-SNN-09: train_epoch() returns (loss, accuracy) as a 2-tuple
    def test_train_epoch_returns_loss_and_accuracy(self, simple_nn_model, small_X_train, small_y_train):
        pytest.skip("Not yet implemented")

    # FR-SNN-10: Accuracy returned by train_epoch() is bounded in [0, 1]
    def test_train_epoch_accuracy_bounded(self, simple_nn_model, small_X_train, small_y_train):
        pytest.skip("Not yet implemented")

    # FR-SNN-11: Loss decreases over multiple training epochs on a learnable problem
    def test_loss_decreases_over_epochs(self, small_X_train, small_y_train):
        pytest.skip("Not yet implemented")


class TestSimpleNNNonFunctional:
    """NFR tests for models.SimpleNN."""

    # NFR-SNN-01: forward() completes within an acceptable time budget
    def test_forward_performance(self, simple_nn_model, small_X_train):
        pytest.skip("Not yet implemented")

    # NFR-SNN-02: Identical seeds produce identical initial weights
    def test_reproducibility_with_seed(self):
        pytest.skip("Not yet implemented")

    # NFR-SNN-03: Model handles a single-sample batch without raising exceptions
    def test_single_sample_input(self, simple_nn_model):
        pytest.skip("Not yet implemented")

    # NFR-SNN-04: Numerical stability — no NaN / Inf values in forward pass output
    def test_no_nan_in_forward_output(self, simple_nn_model, small_X_train):
        pytest.skip("Not yet implemented")


# ===========================================================================
# ComplexNN (2 hidden layers)
# ===========================================================================

class TestComplexNNFunctional:
    """FR tests for models.ComplexNN."""

    # FR-CNN-01: Model initialises with correct shapes for W1, b1, W2, b2, W3, b3
    def test_weight_shapes_on_init(self, complex_nn_model):
        pytest.skip("Not yet implemented")

    # FR-CNN-02: forward() output shape is (n_samples, n_classes)
    def test_forward_output_shape(self, complex_nn_model, small_X_train):
        pytest.skip("Not yet implemented")

    # FR-CNN-03: forward() returns a valid probability distribution
    def test_forward_returns_valid_probabilities(self, complex_nn_model, small_X_train):
        pytest.skip("Not yet implemented")

    # FR-CNN-04: forward() stores intermediate activations for all three layers
    def test_forward_stores_intermediate_activations(self, complex_nn_model, small_X_train):
        pytest.skip("Not yet implemented")

    # FR-CNN-05: compute_loss() returns a non-negative scalar
    def test_compute_loss_is_non_negative(self, complex_nn_model, small_X_train, small_y_train):
        pytest.skip("Not yet implemented")

    # FR-CNN-06: train_epoch() returns (loss, accuracy) as a 2-tuple
    def test_train_epoch_returns_loss_and_accuracy(self, complex_nn_model, small_X_train, small_y_train):
        pytest.skip("Not yet implemented")

    # FR-CNN-07: Accuracy returned by train_epoch() is bounded in [0, 1]
    def test_train_epoch_accuracy_bounded(self, complex_nn_model, small_X_train, small_y_train):
        pytest.skip("Not yet implemented")


class TestComplexNNNonFunctional:
    """NFR tests for models.ComplexNN."""

    # NFR-CNN-01: forward() completes within an acceptable time budget
    def test_forward_performance(self, complex_nn_model, small_X_train):
        pytest.skip("Not yet implemented")

    # NFR-CNN-02: Identical seeds produce identical initial weights
    def test_reproducibility_with_seed(self):
        pytest.skip("Not yet implemented")

    # NFR-CNN-03: No NaN / Inf values appear in the forward pass
    def test_no_nan_in_forward_output(self, complex_nn_model, small_X_train):
        pytest.skip("Not yet implemented")

    # NFR-CNN-04: Model handles a single-sample batch without raising exceptions
    def test_single_sample_input(self, complex_nn_model):
        pytest.skip("Not yet implemented")
