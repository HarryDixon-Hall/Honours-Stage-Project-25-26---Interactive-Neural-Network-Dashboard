import numpy as np
import pytest

from modelFactory.models import ComplexNN, LogisticRegression, SimpleNN
from modelFactory.trainer import build_model, train_model


@pytest.mark.parametrize(
    ("model_name", "expected_type"),
    [
        ("Logistic Regression", LogisticRegression),
        ("simple_nn", SimpleNN),
        ("NN-2-Layer", ComplexNN),
    ],
)
def test_build_model_returns_expected_model_type(model_name, expected_type):
    model = build_model(model_name, input_size=4, output_size=3, hidden_size=5, seed=7)

    assert isinstance(model, expected_type)


def test_build_model_rejects_unknown_model_names():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("unsupported", input_size=4, output_size=3)


def test_train_model_returns_metric_history_for_each_epoch():
    rng = np.random.default_rng(11)
    features = rng.normal(size=(24, 4))
    labels = rng.integers(low=0, high=3, size=24)
    model = build_model("simple_nn", input_size=4, output_size=3, hidden_size=6, seed=11)

    _, history = train_model(model, features, labels, epochs=6, learning_rate=0.01)

    assert len(history["loss"]) == 6
    assert len(history["accuracy"]) == 6
    assert all(isinstance(value, float) for value in history["loss"])


def test_training_updates_model_parameters():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(20, 4))
    labels = rng.integers(low=0, high=3, size=20)
    model = build_model("simple_nn", input_size=4, output_size=3, hidden_size=5, seed=5)
    weights_before = model.W1.copy()

    train_model(model, features, labels, epochs=1, learning_rate=0.05)

    assert not np.allclose(weights_before, model.W1)