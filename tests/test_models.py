import numpy as np
import pytest

from models import LogisticRegression, SimpleNN, ComplexNN

#these are to tests the calculations through each 
# this is for iris dataset in LOGISTIC REGRESSION
def do_classification(batch_size=8, input_size=4, num_classes=3, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(batch_size, input_size)
    y = rng.randint(0, num_classes, size=batch_size)
    return X, y

def test_logistic_forward_shape_and_probs():
    X, y = do_classification()
    model = LogisticRegression(input_size=4, output_size=3, seed=123)

    probs = model.forward(X)

    assert probs.shape == (X.shape[0], 3)
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

def test_logistic_loss_positive_and_finite():
    X, y = do_classification()
    model = LogisticRegression(input_size=4, output_size=3, seed=123)

    probs = model.forward(X)
    loss = model.compute_loss(probs, y)

    assert loss > 0.0
    assert np.isfinite(loss)

def test_logistic_backward_updates_weights():
    X, y = do_classification()
    model = LogisticRegression(input_size=4, output_size=3, seed=123)

    _ = model.forward(X)
    W_before = model.W.copy()
    b_before = model.b.copy()

    model.backward(X, y, learning_rate=0.1)

    assert not np.allclose(model.W, W_before)
    assert not np.allclose(model.b, b_before)

def test_logistic_train_epoch_reduces_loss_over_steps():
    X, y = do_classification(batch_size=64)
    model = LogisticRegression(input_size=4, output_size=3, seed=123)

    losses = []
    for _ in range(10):
        loss, acc = model.train_epoch(X, y, learning_rate=0.1)
        losses.append(loss)
        assert 0.0 <= acc <= 1.0

    assert losses[-1] <= losses[0]

            
#SIMPLE NN TEST
#should i do IRIS dataset
def test_simplenn_forward_shape_and_probs():
    X, y = do_classification()
    model = SimpleNN(input_size=4, hidden_size=8, output_size=3, seed=123)

    probs = model.forward(X)

    assert probs.shape == (X.shape[0], 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

def test_simplenn_loss_positive_and_finite():
    X, y = do_classification()
    model = SimpleNN(input_size=4, hidden_size=8, output_size=3, seed=123)

    probs = model.forward(X)
    loss = model.compute_loss(probs, y)

    assert loss > 0.0
    assert np.isfinite(loss)

def test_simplenn_backward_updates_weights():
    X, y = do_classification()
    model = SimpleNN(input_size=4, hidden_size=8, output_size=3, seed=123)

    _ = model.forward(X)
    W1_before = model.W1.copy()
    b1_before = model.b1.copy()
    W2_before = model.W2.copy()
    b2_before = model.b2.copy()

    model.backward(X, y, learning_rate=0.1)

    assert not np.allclose(model.W1, W1_before)
    assert not np.allclose(model.b1, b1_before)
    assert not np.allclose(model.W2, W2_before)
    assert not np.allclose(model.b2, b2_before)

def test_simplenn_train_epoch_reduces_loss_over_steps():
    X, y = do_classification(batch_size=64)
    model = SimpleNN(input_size=4, hidden_size=8, output_size=3, seed=123)

    losses = []
    for _ in range(10):
        loss, acc = model.train_epoch(X, y, learning_rate=0.1)
        losses.append(loss)
        assert 0.0 <= acc <= 1.0

    assert losses[-1] <= losses[0]

#COMPLEX NN TESTS
#this contains bugs at the moment because the bias is not updating properly after a back pass

def test_complexnn_forward_shape_and_probs():
    X, y = do_classification()
    model = ComplexNN(input_size=4, hidden1_size=8, hidden2_size=8, output_size=3, seed=123)

    probs = model.forward(X)

    assert probs.shape == (X.shape[0], 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

def test_complexnn_loss_positive_and_finite():
    X, y = do_classification()
    model = ComplexNN(input_size=4, hidden1_size=8, hidden2_size=8, output_size=3, seed=123)

    probs = model.forward(X)
    loss = model.compute_loss(probs, y)

    assert loss > 0.0
    assert np.isfinite(loss)
