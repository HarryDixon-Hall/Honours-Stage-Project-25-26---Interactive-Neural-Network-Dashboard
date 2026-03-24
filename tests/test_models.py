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

How pytest achieves the feature → FR/NFR → verification aim
------------------------------------------------------------
Each test function maps directly to one requirement:

  Feature                       FR code     Assertion (what pytest checks)
  ──────────────────────────────────────────────────────────────────────────
  LogisticRegression forward()  FR-LR-03    assert output.shape == (n, 3)
  SimpleNN relu()               FR-SNN-02   assert np.allclose(result, expected)
  ComplexNN weight init         FR-CNN-01   assert model.W3.shape == (8, 3)

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
# LogisticRegression
# ===========================================================================

class TestLogisticRegressionFunctional:
    """FR tests for models.LogisticRegression."""

    # FR-LR-01: Model initialises and weight / bias tensors have the correct shapes
    def test_weight_shapes_on_init(self, logistic_regression_model):
        assert logistic_regression_model.W.shape == (4, 3)
        assert logistic_regression_model.b.shape == (1, 3)

    # FR-LR-02: forward() returns a probability distribution (all values in [0, 1], rows sum to 1)
    def test_forward_returns_valid_probabilities(self, logistic_regression_model, small_X_train):
        probs = logistic_regression_model.forward(small_X_train)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert np.allclose(probs.sum(axis=1), 1.0)

    # FR-LR-03: forward() output shape is (n_samples, n_classes)
    def test_forward_output_shape(self, logistic_regression_model, small_X_train):
        output = logistic_regression_model.forward(small_X_train)
        assert output.shape == (small_X_train.shape[0], 3)

    # FR-LR-04: compute_loss() returns a non-negative scalar
    def test_compute_loss_is_non_negative(self, logistic_regression_model, small_X_train, small_y_train):
        output = logistic_regression_model.forward(small_X_train)
        loss = logistic_regression_model.compute_loss(output, small_y_train)
        assert float(loss) >= 0.0

    # FR-LR-05: backward() updates weights (W and b change after a single backward call)
    def test_backward_updates_weights(self, logistic_regression_model, small_X_train, small_y_train):
        W_before = logistic_regression_model.W.copy()
        b_before = logistic_regression_model.b.copy()
        logistic_regression_model.forward(small_X_train)
        logistic_regression_model.backward(small_X_train, small_y_train, learning_rate=0.1)
        assert not np.array_equal(logistic_regression_model.W, W_before)
        assert not np.array_equal(logistic_regression_model.b, b_before)

    # FR-LR-06: train_epoch() returns (loss, accuracy) as a 2-tuple
    def test_train_epoch_returns_loss_and_accuracy(self, logistic_regression_model, small_X_train, small_y_train):
        result = logistic_regression_model.train_epoch(small_X_train, small_y_train, learning_rate=0.01)
        assert isinstance(result, tuple)
        assert len(result) == 2

    # FR-LR-07: Accuracy returned by train_epoch() is bounded in [0, 1]
    def test_train_epoch_accuracy_bounded(self, logistic_regression_model, small_X_train, small_y_train):
        _, acc = logistic_regression_model.train_epoch(small_X_train, small_y_train, learning_rate=0.01)
        assert 0.0 <= float(acc) <= 1.0

    # FR-LR-08: Loss decreases over multiple training epochs on a learnable problem
    def test_loss_decreases_over_epochs(self, small_X_train, small_y_train):
        from models import LogisticRegression
        model = LogisticRegression(input_size=4, output_size=3, seed=42)
        losses = [model.train_epoch(small_X_train, small_y_train, learning_rate=0.1)[0]
                  for _ in range(200)]
        assert losses[-1] < losses[0]


class TestLogisticRegressionNonFunctional:
    """NFR tests for models.LogisticRegression."""

    # NFR-LR-01: forward() completes within an acceptable time budget for typical input sizes
    def test_forward_performance(self, logistic_regression_model, small_X_train):
        start = time.perf_counter()
        logistic_regression_model.forward(small_X_train)
        assert time.perf_counter() - start < 1.0

    # NFR-LR-02: Using the same seed produces identical weights on separate instances
    def test_reproducibility_with_seed(self):
        from models import LogisticRegression
        m1 = LogisticRegression(input_size=4, output_size=3, seed=99)
        m2 = LogisticRegression(input_size=4, output_size=3, seed=99)
        assert np.array_equal(m1.W, m2.W)
        assert np.array_equal(m1.b, m2.b)

    # NFR-LR-03: Model handles edge-case input (single sample) without raising an exception
    def test_single_sample_input(self, logistic_regression_model):
        X_single = np.random.randn(1, 4)
        output = logistic_regression_model.forward(X_single)
        assert output.shape == (1, 3)

    # NFR-LR-04: Model handles large input batches without memory errors
    def test_large_batch_input(self, logistic_regression_model):
        X_large = np.random.randn(10_000, 4)
        output = logistic_regression_model.forward(X_large)
        assert output.shape == (10_000, 3)


# ===========================================================================
# SimpleNN (1 hidden layer)
# ===========================================================================

class TestSimpleNNFunctional:
    """FR tests for models.SimpleNN."""

    # FR-SNN-01: Model initialises with correct shapes for W1, b1, W2, b2
    def test_weight_shapes_on_init(self, simple_nn_model):
        assert simple_nn_model.W1.shape == (4, 8)
        assert simple_nn_model.b1.shape == (1, 8)
        assert simple_nn_model.W2.shape == (8, 3)
        assert simple_nn_model.b2.shape == (1, 3)

    # FR-SNN-02: relu() returns zero for negative inputs and is identity for positive inputs
    def test_relu_activation(self, simple_nn_model):
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = simple_nn_model.relu(x)
        assert np.allclose(result, [0.0, 0.0, 0.0, 0.5, 2.0])

    # FR-SNN-03: relu_derivative() returns 0 for x <= 0 and 1 for x > 0
    def test_relu_derivative(self, simple_nn_model):
        x = np.array([-1.0, 0.0, 1.0])
        result = simple_nn_model.relu_derivative(x)
        assert np.allclose(result, [0.0, 0.0, 1.0])

    # FR-SNN-04: softmax() output rows sum to 1 and all values are in (0, 1)
    def test_softmax_output(self, simple_nn_model):
        x = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 1.5]])
        result = simple_nn_model.softmax(x)
        assert np.all(result > 0.0)
        assert np.all(result < 1.0)
        assert np.allclose(result.sum(axis=1), 1.0)

    # FR-SNN-05: forward() stores intermediate activations (a1, z1, a2, z2) as instance attributes
    def test_forward_stores_intermediate_activations(self, simple_nn_model, small_X_train):
        simple_nn_model.forward(small_X_train)
        assert hasattr(simple_nn_model, 'z1')
        assert hasattr(simple_nn_model, 'a1')
        assert hasattr(simple_nn_model, 'z2')
        assert hasattr(simple_nn_model, 'a2')

    # FR-SNN-06: forward() output shape is (n_samples, n_classes)
    def test_forward_output_shape(self, simple_nn_model, small_X_train):
        output = simple_nn_model.forward(small_X_train)
        assert output.shape == (small_X_train.shape[0], 3)

    # FR-SNN-07: compute_loss() returns a non-negative scalar
    def test_compute_loss_is_non_negative(self, simple_nn_model, small_X_train, small_y_train):
        output = simple_nn_model.forward(small_X_train)
        loss = simple_nn_model.compute_loss(output, small_y_train)
        assert float(loss) >= 0.0

    # FR-SNN-08: backward() updates all four parameter tensors (W1, b1, W2, b2)
    def test_backward_updates_all_weights(self, simple_nn_model, small_X_train, small_y_train):
        W1_before = simple_nn_model.W1.copy()
        b1_before = simple_nn_model.b1.copy()
        W2_before = simple_nn_model.W2.copy()
        b2_before = simple_nn_model.b2.copy()
        simple_nn_model.forward(small_X_train)
        simple_nn_model.backward(small_X_train, small_y_train, learning_rate=0.1)
        assert not np.array_equal(simple_nn_model.W1, W1_before)
        assert not np.array_equal(simple_nn_model.b1, b1_before)
        assert not np.array_equal(simple_nn_model.W2, W2_before)
        assert not np.array_equal(simple_nn_model.b2, b2_before)

    # FR-SNN-09: train_epoch() returns (loss, accuracy) as a 2-tuple
    def test_train_epoch_returns_loss_and_accuracy(self, simple_nn_model, small_X_train, small_y_train):
        result = simple_nn_model.train_epoch(small_X_train, small_y_train, learning_rate=0.01)
        assert isinstance(result, tuple)
        assert len(result) == 2

    # FR-SNN-10: Accuracy returned by train_epoch() is bounded in [0, 1]
    def test_train_epoch_accuracy_bounded(self, simple_nn_model, small_X_train, small_y_train):
        _, acc = simple_nn_model.train_epoch(small_X_train, small_y_train, learning_rate=0.01)
        assert 0.0 <= float(acc) <= 1.0

    # FR-SNN-11: Loss decreases over multiple training epochs on a learnable problem
    def test_loss_decreases_over_epochs(self, small_X_train, small_y_train):
        from models import SimpleNN
        model = SimpleNN(input_size=4, hidden_size=8, output_size=3, seed=42)
        losses = [model.train_epoch(small_X_train, small_y_train, learning_rate=0.1)[0]
                  for _ in range(200)]
        assert losses[-1] < losses[0]


class TestSimpleNNNonFunctional:
    """NFR tests for models.SimpleNN."""

    # NFR-SNN-01: forward() completes within an acceptable time budget
    def test_forward_performance(self, simple_nn_model, small_X_train):
        start = time.perf_counter()
        simple_nn_model.forward(small_X_train)
        assert time.perf_counter() - start < 1.0

    # NFR-SNN-02: Identical seeds produce identical initial weights
    def test_reproducibility_with_seed(self):
        from models import SimpleNN
        m1 = SimpleNN(input_size=4, hidden_size=8, output_size=3, seed=7)
        m2 = SimpleNN(input_size=4, hidden_size=8, output_size=3, seed=7)
        assert np.array_equal(m1.W1, m2.W1)
        assert np.array_equal(m1.W2, m2.W2)

    # NFR-SNN-03: Model handles a single-sample batch without raising exceptions
    def test_single_sample_input(self, simple_nn_model):
        X_single = np.random.randn(1, 4)
        output = simple_nn_model.forward(X_single)
        assert output.shape == (1, 3)

    # NFR-SNN-04: Numerical stability — no NaN / Inf values in forward pass output
    def test_no_nan_in_forward_output(self, simple_nn_model, small_X_train):
        output = simple_nn_model.forward(small_X_train)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))


# ===========================================================================
# ComplexNN (2 hidden layers)
# ===========================================================================

class TestComplexNNFunctional:
    """FR tests for models.ComplexNN."""

    # FR-CNN-01: Model initialises with correct shapes for W1, b1, W2, b2, W3, b3
    def test_weight_shapes_on_init(self, complex_nn_model):
        assert complex_nn_model.W1.shape == (4, 8)
        assert complex_nn_model.b1.shape == (1, 8)
        assert complex_nn_model.W2.shape == (8, 8)
        assert complex_nn_model.b2.shape == (1, 8)
        assert complex_nn_model.W3.shape == (8, 3)
        assert complex_nn_model.b3.shape == (1, 3)

    # FR-CNN-02: forward() output shape is (n_samples, n_classes)
    def test_forward_output_shape(self, complex_nn_model, small_X_train):
        output = complex_nn_model.forward(small_X_train)
        assert output.shape == (small_X_train.shape[0], 3)

    # FR-CNN-03: forward() returns a valid probability distribution
    def test_forward_returns_valid_probabilities(self, complex_nn_model, small_X_train):
        probs = complex_nn_model.forward(small_X_train)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert np.allclose(probs.sum(axis=1), 1.0)

    # FR-CNN-04: forward() stores intermediate activations for all three layers
    def test_forward_stores_intermediate_activations(self, complex_nn_model, small_X_train):
        complex_nn_model.forward(small_X_train)
        for attr in ('z1', 'a1', 'z2', 'a2', 'z3', 'a3'):
            assert hasattr(complex_nn_model, attr)

    # FR-CNN-05: compute_loss() returns a non-negative scalar
    def test_compute_loss_is_non_negative(self, complex_nn_model, small_X_train, small_y_train):
        output = complex_nn_model.forward(small_X_train)
        loss = complex_nn_model.compute_loss(output, small_y_train)
        assert float(loss) >= 0.0

    # FR-CNN-06: train_epoch() returns (loss, accuracy) as a 2-tuple
    def test_train_epoch_returns_loss_and_accuracy(self, complex_nn_model, small_X_train, small_y_train):
        result = complex_nn_model.train_epoch(small_X_train, small_y_train, learning_rate=0.01)
        assert isinstance(result, tuple)
        assert len(result) == 2

    # FR-CNN-07: Accuracy returned by train_epoch() is bounded in [0, 1]
    def test_train_epoch_accuracy_bounded(self, complex_nn_model, small_X_train, small_y_train):
        _, acc = complex_nn_model.train_epoch(small_X_train, small_y_train, learning_rate=0.01)
        assert 0.0 <= float(acc) <= 1.0


class TestComplexNNNonFunctional:
    """NFR tests for models.ComplexNN."""

    # NFR-CNN-01: forward() completes within an acceptable time budget
    def test_forward_performance(self, complex_nn_model, small_X_train):
        start = time.perf_counter()
        complex_nn_model.forward(small_X_train)
        assert time.perf_counter() - start < 1.0

    # NFR-CNN-02: Identical seeds produce identical initial weights
    def test_reproducibility_with_seed(self):
        from models import ComplexNN
        m1 = ComplexNN(input_size=4, hidden1_size=8, hidden2_size=8, output_size=3, seed=7)
        m2 = ComplexNN(input_size=4, hidden1_size=8, hidden2_size=8, output_size=3, seed=7)
        assert np.array_equal(m1.W1, m2.W1)
        assert np.array_equal(m1.W2, m2.W2)
        assert np.array_equal(m1.W3, m2.W3)

    # NFR-CNN-03: No NaN / Inf values appear in the forward pass
    def test_no_nan_in_forward_output(self, complex_nn_model, small_X_train):
        output = complex_nn_model.forward(small_X_train)
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    # NFR-CNN-04: Model handles a single-sample batch without raising exceptions
    def test_single_sample_input(self, complex_nn_model):
        X_single = np.random.randn(1, 4)
        output = complex_nn_model.forward(X_single)
        assert output.shape == (1, 3)

