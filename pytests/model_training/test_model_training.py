import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pytests.helpers import collect_component_ids

from modelFactory.trainer import build_model, train_model
from pages.levels.level1.layout import level1_layout
from pages.levels.level2.layout import level2_layout
from pages.levels.level2.methods import (
    apply_level2_gradients,
    build_level2_dataset,
    compute_level2_gradients,
    init_level2_mlp,
    level2_evaluate_metrics,
    level2_forward_pass,
    level2_set_baseline_history,
    make_activation_figure,
    make_decision_boundary_figure,
    make_level2_training_curves_figure,
    make_network_diagram_figure,
    train_level2_model,
)
from pages.levels.level3.layout import level3_layout
from pages.levels.level3.methods import (
    level3_build_meta,
    level3_extract_meta_from_code,
    level3_initialise_model,
    level3_initialise_store,
)


# ---------------------------------------------------------------------------
# AT-2.1.1.1
# F-2.1.1 — Preconfigured model can be loaded, explored, and trained
# ---------------------------------------------------------------------------

def test_at_2_1_1_1_preconfigured_model_can_be_loaded_explored_and_trained():
    # Step 1: ASSERT a preconfigured model can be loaded
    model = build_model("simple_nn", input_size=4, output_size=3, hidden_size=8, seed=1)
    assert model is not None, "Step 1: preconfigured model must be loadable via build_model"

    # Step 2: ASSERT the loaded model can be explored through the interface
    assert hasattr(model, "W1"), "Step 2: model must expose first-layer weight matrix"
    assert hasattr(model, "W2"), "Step 2: model must expose second-layer weight matrix"
    assert model.W1.shape == (4, 8), "Step 2: W1 shape must reflect declared input and hidden dimensions"

    # Step 3: ASSERT the loaded model can be trained
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4))
    y = rng.integers(0, 3, size=30)
    trained_model, history = train_model(model, X, y, epochs=5)
    assert trained_model is not None, "Step 3: training must return a model"
    assert len(history["loss"]) == 5, "Step 3: training must produce one loss entry per epoch"


# ---------------------------------------------------------------------------
# AT-2.1.1.2
# F-2.1.1 / F-2.1.2 / F-2.1.3 / F-2.1.4 — Training executes for more than
# one epoch across model types and feature paths
# ---------------------------------------------------------------------------

def test_at_2_1_1_2_model_training_executes_for_multiple_epochs():
    # Step 1: ASSERT model training executes for more than one epoch (modelFactory path)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 4))
    y = rng.integers(0, 3, size=30)
    model = build_model("simple_nn", input_size=4, output_size=3, hidden_size=6, seed=2)
    _, history = train_model(model, X, y, epochs=5)
    assert len(history["loss"]) > 1, "Step 1: modelFactory training must record more than one epoch"

    # Step 2: ASSERT the level-2 training pipeline also runs for multiple epochs
    dataset = build_level2_dataset("moons", input_dim=2)
    params = init_level2_mlp(input_dim=2, hidden_layers=[6, 6], output_dim=1, rng=np.random.default_rng(2))
    params = level2_set_baseline_history(dataset, params, "tanh")
    params = train_level2_model(dataset, params, activation="tanh", epochs=5, lr=0.05)
    assert params["epoch"] > 1, "Step 2: level-2 training pipeline must execute more than one epoch"


# ---------------------------------------------------------------------------
# AT-2.1.1.3
# F-2.1.1 / F-2.1.4 / F-2.1.5 — Training loss and accuracy are produced
# and updated during training
# ---------------------------------------------------------------------------

def test_at_2_1_1_3_training_loss_and_accuracy_produced():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 4))
    y = rng.integers(0, 3, size=30)
    model = build_model("simple_nn", input_size=4, output_size=3, hidden_size=6, seed=3)
    _, history = train_model(model, X, y, epochs=5)

    # Step 1: ASSERT training loss is displayed
    assert "loss" in history, "Step 1: training history must contain 'loss'"
    assert all(isinstance(v, float) for v in history["loss"]), "Step 1: each loss value must be a float"

    # Step 2: ASSERT training accuracy is displayed
    assert "accuracy" in history, "Step 2: training history must contain 'accuracy'"
    assert all(isinstance(v, float) for v in history["accuracy"]), "Step 2: each accuracy value must be a float"

    # Step 3: ASSERT training metrics are updated during training
    dataset = build_level2_dataset("moons", input_dim=2)
    params = init_level2_mlp(input_dim=2, hidden_layers=[6, 6], output_dim=1, rng=np.random.default_rng(3))
    params = level2_set_baseline_history(dataset, params, "tanh")
    params = train_level2_model(dataset, params, activation="tanh", epochs=5, lr=0.05)
    assert len(params["history"]["train_loss"]) > 0, "Step 3: train_loss history must be populated"
    assert len(params["history"]["train_accuracy"]) > 0, "Step 3: train_accuracy history must be populated"


# ---------------------------------------------------------------------------
# AT-2.1.1.4
# F-2.1.1 / F-2.1.2 / F-2.1.5 — All four required visualisations exist
# ---------------------------------------------------------------------------

def test_at_2_1_1_4_required_visualisations_exist():
    dataset = build_level2_dataset("moons", input_dim=2)
    params = init_level2_mlp(input_dim=2, hidden_layers=[6, 6], output_dim=1, rng=np.random.default_rng(4))
    params = level2_set_baseline_history(dataset, params, "tanh")

    # Step 1: ASSERT a model architecture visualisation exists
    arch_fig = make_network_diagram_figure(
        input_dim=2, hidden_layers=[6, 6], output_dim=1, params=params
    )
    assert arch_fig is not None, "Step 1: architecture visualisation must be produced"
    assert len(arch_fig.data) > 0, "Step 1: architecture figure must contain at least one trace"

    # Step 2: ASSERT a decision boundary visualisation exists
    boundary_fig = make_decision_boundary_figure(dataset, params, "tanh")
    assert boundary_fig is not None, "Step 2: decision boundary visualisation must be produced"
    assert len(boundary_fig.data) > 0, "Step 2: decision boundary figure must contain at least one trace"

    # Step 3: ASSERT an activation visualisation exists
    activation_fig = make_activation_figure("tanh")
    assert activation_fig is not None, "Step 3: activation visualisation must be produced"
    assert len(activation_fig.data) > 0, "Step 3: activation figure must contain at least one trace"

    # Step 4: ASSERT a training curve visualisation exists
    params = train_level2_model(dataset, params, activation="tanh", epochs=5, lr=0.05)
    curve_fig = make_level2_training_curves_figure(params["history"])
    assert curve_fig is not None, "Step 4: training curve visualisation must be produced"
    assert len(curve_fig.data) > 0, "Step 4: training curve figure must contain at least one trace"


# ---------------------------------------------------------------------------
# AT-2.1.1.5
# F-2.1.1 / F-2.1.3 / F-2.1.4 / F-2.1.5 — Trained model can be evaluated
# on held-out data and evaluation output is produced
# ---------------------------------------------------------------------------

def test_at_2_1_1_5_trained_model_evaluated_on_held_out_data():
    dataset = build_level2_dataset("moons", input_dim=2)
    params = init_level2_mlp(input_dim=2, hidden_layers=[6, 6], output_dim=1, rng=np.random.default_rng(5))
    params = level2_set_baseline_history(dataset, params, "tanh")
    params = train_level2_model(dataset, params, activation="tanh", epochs=5, lr=0.05)

    # Step 1: ASSERT the trained model can be evaluated on held-out data
    metrics = level2_evaluate_metrics(dataset, params, "tanh")

    # Step 2: ASSERT evaluation output is produced
    assert "test_loss" in metrics, "Step 2: evaluation must produce 'test_loss'"
    assert "test_accuracy" in metrics, "Step 2: evaluation must produce 'test_accuracy'"
    assert isinstance(metrics["test_loss"], float), "Step 2: test_loss must be a float"
    assert isinstance(metrics["test_accuracy"], float), "Step 2: test_accuracy must be a float"
    assert 0.0 <= metrics["test_accuracy"] <= 1.0, "Step 2: test_accuracy must be in [0, 1]"


# ---------------------------------------------------------------------------
# AT-2.1.2.1
# F-2.1.2 — Model construction controls are present in the UI and a
# configuration can be submitted
# ---------------------------------------------------------------------------

def test_at_2_1_2_1_model_construction_controls_present_in_ui():
    ids = collect_component_ids(level2_layout())

    # Step 1: ASSERT model construction controls are present in the UI
    assert "level2-train-toggle-btn" in ids, "Step 1: 'Start Training' button must be present"
    assert "level2-hidden-layers-slider" in ids, "Step 1: hidden-layer count control must be present"
    assert "level2-activation-dropdown" in ids, "Step 1: activation function control must be present"

    # Step 2: ASSERT a user can submit a model configuration through the UI
    assert "level2-input-dim-input" in ids, "Step 2: input dimension control must be present"
    assert "level2-output-dim-input" in ids, "Step 2: output dimension control must be present"


# ---------------------------------------------------------------------------
# AT-2.1.2.2
# F-2.1.2 / F-2.1.3 — All architecture parameter controls exist
# ---------------------------------------------------------------------------

def test_at_2_1_2_2_architecture_parameter_controls_exist():
    ids = collect_component_ids(level2_layout())

    # Step 1: ASSERT input size control exists
    assert "level2-input-dim-input" in ids, "Step 1: input size control must exist"

    # Step 2: ASSERT hidden layer count control exists
    assert "level2-hidden-layers-slider" in ids, "Step 2: hidden layer count control must exist"

    # Step 3: ASSERT neuron count control exists
    assert "level2-hidden-layer-1-input" in ids, "Step 3: per-layer neuron count control must exist"

    # Step 4: ASSERT activation function control exists
    assert "level2-activation-dropdown" in ids, "Step 4: activation function control must exist"

    # Step 5: ASSERT output structure control exists
    assert "level2-output-dim-input" in ids, "Step 5: output structure control must exist"


# ---------------------------------------------------------------------------
# AT-2.1.2.3
# F-2.1.2 / F-2.1.3 — A valid model is initialised from a selected
# UI configuration and from submitted code
# ---------------------------------------------------------------------------

def test_at_2_1_2_3_valid_model_initialised_from_configuration():
    # Step 1: ASSERT a valid model is initialised from a selected UI configuration
    params = init_level2_mlp(
        input_dim=2,
        hidden_layers=[4, 4],
        output_dim=1,
        rng=np.random.default_rng(7),
    )
    assert params is not None, "Step 1: model must be returned from UI-driven init"
    assert len(params["weights"]) == 3, "Step 1: model must have weight matrices for input→H1, H1→H2, H2→output"
    assert len(params["biases"]) == 3, "Step 1: model must have bias vectors for every layer transition"

    # Step 2: ASSERT a valid model is initialised from submitted code (F-2.1.3 path)
    meta = level3_build_meta(
        dataset="moons",
        hidden_layer_sizes=[6, 4],
        activation="tanh",
        epochs=10,
        learning_rate=0.05,
        input_dim=2,
        output_dim=1,
    )
    store = level3_initialise_store(meta)
    store = level3_initialise_model(store, meta)
    assert store["model"] is not None, "Step 2: code-path init must produce a model"
    assert len(store["model"]["weights"]) > 0, "Step 2: code-defined model must contain weight matrices"


# ---------------------------------------------------------------------------
# AT-2.1.3.1
# F-2.1.3 — User can submit model structure code and it is parsed into
# a model definition
# ---------------------------------------------------------------------------

def test_at_2_1_3_1_user_can_submit_model_structure_code():
    ids = collect_component_ids(level3_layout())

    # Step 1: ASSERT a user can submit model structure code
    for cell_number in range(1, 7):
        assert f"level3-cell-{cell_number}-code" in ids, (
            f"Step 1: code cell {cell_number} input must be present in the level-3 layout"
        )

    # Step 2: ASSERT submitted code is parsed into a model definition
    cell_1 = 'dataset_name = "moons"'
    cell_2 = "input_dim = 2"
    cell_3 = 'activation = "tanh"'
    cell_4 = "hidden_layers = [6, 4]"
    cell_5 = "output_dim = 1"
    cell_6 = "epochs = 10\nlearning_rate = 0.05"
    meta = level3_extract_meta_from_code(cell_1, cell_2, cell_3, cell_4, cell_5, cell_6)
    assert meta["dataset"] == "moons", "Step 2: dataset_name must be parsed from code"
    assert meta["input_dim"] == 2, "Step 2: input_dim must be parsed from code"
    assert meta["activation"] == "tanh", "Step 2: activation must be parsed from code"
    assert meta["hidden_layer_sizes"] == [6, 4], "Step 2: hidden layer structure must be parsed from code"


# ---------------------------------------------------------------------------
# AT-2.1.4.1
# F-2.1.4 — All four stages of the training pipeline are executed:
# forward pass, loss computation, backpropagation, parameter update
# ---------------------------------------------------------------------------

def test_at_2_1_4_1_training_pipeline_stages_executed():
    dataset = build_level2_dataset("moons", input_dim=2)
    params = init_level2_mlp(input_dim=2, hidden_layers=[6, 6], output_dim=1, rng=np.random.default_rng(41))
    params = level2_set_baseline_history(dataset, params, "tanh")

    # Step 1: ASSERT forward-pass stage is executed
    predictions, cache = level2_forward_pass(dataset["X_train"], params, "tanh")
    assert predictions is not None, "Step 1: forward pass must return predictions"
    assert predictions.shape[1] == dataset["X_train"].shape[0], (
        "Step 1: prediction column count must match the batch size"
    )

    # Step 2: ASSERT loss computation stage is executed
    gradient_snapshot = compute_level2_gradients(dataset, params, activation="tanh")
    assert "loss" in gradient_snapshot, "Step 2: gradient computation must include a loss value"
    assert isinstance(gradient_snapshot["loss"], float), "Step 2: loss must be a float"

    # Step 3: ASSERT backpropagation stage is executed
    assert "weight_gradients" in gradient_snapshot, "Step 3: backprop must produce weight gradients"
    assert "gradient_norms" in gradient_snapshot, "Step 3: backprop must produce gradient norms"
    assert len(gradient_snapshot["weight_gradients"]) == len(params["weights"]), (
        "Step 3: one gradient matrix must exist per weight layer"
    )

    # Step 4: ASSERT parameter-update stage is executed
    weights_before = [list(row) for layer in params["weights"] for row in layer]
    updated_params = apply_level2_gradients(dataset, params, gradient_snapshot, activation="tanh", lr=0.05)
    weights_after = [list(row) for layer in updated_params["weights"] for row in layer]
    assert weights_before != weights_after, "Step 4: parameter update must change at least one weight"