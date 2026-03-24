#DECLARATION: CODE HERE IS ASSISTED BY https://www.perplexity.ai/ 04/12/25 - 22/03/26
"""
test_trainer.py — Test backbone for trainer.py
===============================================
Covers the two public functions in trainer.py:
  • build_model()  — model factory
  • train_model()  — training loop

Each class is split into FR (Functional Requirements) and NFR (Non-Functional Requirements).

How pytest achieves the feature → FR/NFR → verification aim
------------------------------------------------------------
Each test function maps directly to one requirement:

  Feature                       FR code     Assertion (what pytest checks)
  ──────────────────────────────────────────────────────────────────────────
  build_model factory           FR-BM-01    assert isinstance(model, LogisticRegression)
  build_model unknown name      FR-BM-04    with pytest.raises(ValueError): ...
  train_model history keys      FR-TM-02    assert 'loss' in history

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
# build_model()
# ===========================================================================

class TestBuildModelFunctional:
    """FR tests for trainer.build_model()."""

    # FR-BM-01: build_model("Logistic Regression", ...) returns a LogisticRegression instance
    def test_builds_logistic_regression(self):
        from trainer import build_model
        from models import LogisticRegression
        model = build_model("Logistic Regression", input_size=4, output_size=3)
        assert isinstance(model, LogisticRegression)

    # FR-BM-02: build_model("simple_nn", ...) returns a SimpleNN instance
    def test_builds_simple_nn(self):
        from trainer import build_model
        from models import SimpleNN
        model = build_model("simple_nn", input_size=4, output_size=3, hidden_size=8)
        assert isinstance(model, SimpleNN)

    # FR-BM-03: build_model("NN-2-Layer", ...) returns a ComplexNN instance
    def test_builds_complex_nn(self):
        from trainer import build_model
        from models import ComplexNN
        model = build_model("NN-2-Layer", input_size=4, output_size=3, hidden_size=8)
        assert isinstance(model, ComplexNN)

    # FR-BM-04: build_model() raises ValueError for an unrecognised model name
    def test_raises_for_unknown_model(self):
        from trainer import build_model
        with pytest.raises(ValueError):
            build_model("not_a_model", input_size=4, output_size=3)

    # FR-BM-05: Returned LogisticRegression has weight tensors matching input/output sizes
    def test_logistic_regression_weight_shapes(self):
        from trainer import build_model
        model = build_model("Logistic Regression", input_size=6, output_size=4)
        assert model.W.shape == (6, 4)
        assert model.b.shape == (1, 4)

    # FR-BM-06: Returned SimpleNN has weight tensors matching input/hidden/output sizes
    def test_simple_nn_weight_shapes(self):
        from trainer import build_model
        model = build_model("simple_nn", input_size=6, output_size=4, hidden_size=10)
        assert model.W1.shape == (6, 10)
        assert model.W2.shape == (10, 4)

    # FR-BM-07: Returned ComplexNN has weight tensors matching input/hidden1/hidden2/output sizes
    def test_complex_nn_weight_shapes(self):
        from trainer import build_model
        model = build_model("NN-2-Layer", input_size=6, output_size=4, hidden_size=10)
        assert model.W1.shape == (6, 10)
        assert model.W2.shape == (10, 10)
        assert model.W3.shape == (10, 4)

    # FR-BM-08: The seed parameter controls weight initialisation reproducibly
    def test_seed_produces_reproducible_weights(self):
        from trainer import build_model
        m1 = build_model("simple_nn", input_size=4, output_size=3, hidden_size=8, seed=123)
        m2 = build_model("simple_nn", input_size=4, output_size=3, hidden_size=8, seed=123)
        assert np.array_equal(m1.W1, m2.W1)
        assert np.array_equal(m1.W2, m2.W2)


class TestBuildModelNonFunctional:
    """NFR tests for trainer.build_model()."""

    # NFR-BM-01: build_model() completes within an acceptable time for all supported model types
    def test_build_performance_logistic_regression(self):
        from trainer import build_model
        start = time.perf_counter()
        build_model("Logistic Regression", input_size=4, output_size=3)
        assert time.perf_counter() - start < 1.0

    # NFR-BM-02: build_model() completes within an acceptable time for SimpleNN
    def test_build_performance_simple_nn(self):
        from trainer import build_model
        start = time.perf_counter()
        build_model("simple_nn", input_size=4, output_size=3, hidden_size=8)
        assert time.perf_counter() - start < 1.0

    # NFR-BM-03: build_model() completes within an acceptable time for ComplexNN
    def test_build_performance_complex_nn(self):
        from trainer import build_model
        start = time.perf_counter()
        build_model("NN-2-Layer", input_size=4, output_size=3, hidden_size=8)
        assert time.perf_counter() - start < 1.0


# ===========================================================================
# train_model()
# ===========================================================================

class TestTrainModelFunctional:
    """FR tests for trainer.train_model()."""

    # FR-TM-01: train_model() returns a 2-tuple of (trained_model, history)
    def test_returns_model_and_history(self, built_logistic_regression, small_X_train, small_y_train):
        from trainer import train_model
        result = train_model(built_logistic_regression, small_X_train, small_y_train, epochs=5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    # FR-TM-02: history dict contains the keys 'loss' and 'accuracy'
    def test_history_has_loss_and_accuracy_keys(self, built_logistic_regression, small_X_train, small_y_train):
        from trainer import train_model
        _, history = train_model(built_logistic_regression, small_X_train, small_y_train, epochs=5)
        assert 'loss' in history
        assert 'accuracy' in history

    # FR-TM-03: Length of history['loss'] equals the number of requested epochs
    def test_history_loss_length_equals_epochs(self, built_logistic_regression, small_X_train, small_y_train):
        from trainer import train_model
        _, history = train_model(built_logistic_regression, small_X_train, small_y_train, epochs=10)
        assert len(history['loss']) == 10

    # FR-TM-04: Length of history['accuracy'] equals the number of requested epochs
    def test_history_accuracy_length_equals_epochs(self, built_logistic_regression, small_X_train, small_y_train):
        from trainer import train_model
        _, history = train_model(built_logistic_regression, small_X_train, small_y_train, epochs=10)
        assert len(history['accuracy']) == 10

    # FR-TM-05: All recorded losses are non-negative scalars
    def test_all_losses_are_non_negative(self, built_logistic_regression, small_X_train, small_y_train):
        from trainer import train_model
        _, history = train_model(built_logistic_regression, small_X_train, small_y_train, epochs=10)
        assert all(loss >= 0 for loss in history['loss'])

    # FR-TM-06: All recorded accuracies are in [0, 1]
    def test_all_accuracies_are_bounded(self, built_logistic_regression, small_X_train, small_y_train):
        from trainer import train_model
        _, history = train_model(built_logistic_regression, small_X_train, small_y_train, epochs=10)
        assert all(0.0 <= acc <= 1.0 for acc in history['accuracy'])

    # FR-TM-07: Model weights change after training (the model actually learns)
    def test_weights_change_after_training(self, built_logistic_regression, small_X_train, small_y_train):
        from trainer import train_model
        W_before = built_logistic_regression.W.copy()
        train_model(built_logistic_regression, small_X_train, small_y_train, epochs=5)
        assert not np.array_equal(built_logistic_regression.W, W_before)

    # FR-TM-08: train_model() works correctly with SimpleNN as the model argument
    def test_trains_simple_nn(self, built_simple_nn, small_X_train, small_y_train):
        from trainer import train_model
        _, history = train_model(built_simple_nn, small_X_train, small_y_train, epochs=5)
        assert len(history['loss']) == 5

    # FR-TM-09: Requesting 1 epoch produces exactly one entry in history
    def test_single_epoch_history_length(self, built_logistic_regression, small_X_train, small_y_train):
        from trainer import train_model
        _, history = train_model(built_logistic_regression, small_X_train, small_y_train, epochs=1)
        assert len(history['loss']) == 1
        assert len(history['accuracy']) == 1

    # FR-TM-10: Custom learning_rate is applied (history differs from default rate)
    def test_custom_learning_rate_affects_training(self, small_X_train, small_y_train):
        from trainer import build_model, train_model
        m1 = build_model("Logistic Regression", input_size=4, output_size=3, seed=42)
        m2 = build_model("Logistic Regression", input_size=4, output_size=3, seed=42)
        _, h1 = train_model(m1, small_X_train, small_y_train, epochs=5, learning_rate=0.001)
        _, h2 = train_model(m2, small_X_train, small_y_train, epochs=5, learning_rate=0.5)
        assert h1['loss'] != h2['loss']


class TestTrainModelNonFunctional:
    """NFR tests for trainer.train_model()."""

    # NFR-TM-01: Training 50 epochs on a small dataset completes within an acceptable time
    def test_training_performance_50_epochs(self, built_logistic_regression, small_X_train, small_y_train):
        from trainer import train_model
        start = time.perf_counter()
        train_model(built_logistic_regression, small_X_train, small_y_train, epochs=50)
        assert time.perf_counter() - start < 5.0

    # NFR-TM-02: No NaN / Inf values appear in the loss history after training
    def test_no_nan_in_loss_history(self, built_logistic_regression, small_X_train, small_y_train):
        from trainer import train_model
        _, history = train_model(built_logistic_regression, small_X_train, small_y_train, epochs=50)
        assert all(np.isfinite(loss) for loss in history['loss'])

    # NFR-TM-03: No NaN / Inf values appear in the accuracy history after training
    def test_no_nan_in_accuracy_history(self, built_logistic_regression, small_X_train, small_y_train):
        from trainer import train_model
        _, history = train_model(built_logistic_regression, small_X_train, small_y_train, epochs=50)
        assert all(np.isfinite(acc) for acc in history['accuracy'])

    # NFR-TM-04: train_model() does not raise for the minimum viable epoch count (1)
    def test_minimum_epochs_does_not_raise(self, built_logistic_regression, small_X_train, small_y_train):
        from trainer import train_model
        train_model(built_logistic_regression, small_X_train, small_y_train, epochs=1)

