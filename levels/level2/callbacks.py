from importlib import import_module
from sre_parse import State

import dash
from dash import html
try:
    from dash import Input, Output
except ImportError:
    dash_dependencies = import_module('dash.dependencies')
    Input = dash_dependencies.Input
    Output = dash_dependencies.Output
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.decomposition import PCA

from dataload import load_dataset
from models import SimpleNN

#====DECLARATION: CODE HERE IS ASSISTED BY Copilot (GPT-5.4) 22/03/26 - 23/04/26====
#region MODEL FACTORY: Level 2 callbacks/methods to visual the structure of a single hidde layer perception with decision boundary - WORK IN PROGRESS
#toy datasets 
def load_toy_dataset(name, n_samples=300, noise=0.2, random_state=0):
    if name == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif name == 'circles':
        X, y = make_circles(n_samples=n_samples, factor=0.5,
                            noise=noise, random_state=random_state)
    elif name == 'linear':
        # simple linearly separable classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=1.5,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Ensure shapes (N, 2) and (N,)
    return X.astype(np.float32), y.astype(np.int32)

def init_level2_mlp(input_dim=2, hidden_layers=None, output_dim=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if hidden_layers is None:
        hidden_layers = [6, 6]

    layer_sizes = [input_dim] + list(hidden_layers) + [output_dim]
    weights = []
    biases = []

    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        weights.append(
            rng.normal(0.0, 1.0 / np.sqrt(fan_in), size=(fan_out, fan_in))
        )
        biases.append(np.zeros((fan_out, 1)))

    return {
        'weights': [weight.tolist() for weight in weights],
        'biases': [bias.tolist() for bias in biases],
        'epoch': 0,
        'history': {
            'epochs': [0],
            'loss': [],
            'accuracy': [],
        },
    }


def level2_deserialize_params(params):
    weights = [np.array(weight, dtype=np.float64) for weight in params['weights']]
    biases = [np.array(bias, dtype=np.float64) for bias in params['biases']]
    return weights, biases


def level2_serialize_params(weights, biases, params):
    params['weights'] = [weight.tolist() for weight in weights]
    params['biases'] = [bias.tolist() for bias in biases]
    return params


def level2_parameter_count(params):
    weights, biases = level2_deserialize_params(params)
    return int(sum(weight.size + bias.size for weight, bias in zip(weights, biases)))

def activation_forward(Z, activation):
    if activation == 'relu':
        A = np.maximum(0, Z)
    elif activation == 'tanh':
        A = np.tanh(Z)
    elif activation == 'sigmoid':
        A = 1.0 / (1.0 + np.exp(-Z))
    else:
        raise ValueError(f"Unknown activation: {activation}")
    return A

def activation_backward(dA, Z, activation):
    if activation == 'relu':
        dZ = dA * (Z > 0)
    elif activation == 'tanh':
        A = np.tanh(Z)
        dZ = dA * (1 - A**2)
    elif activation == 'sigmoid':
        A = 1.0 / (1.0 + np.exp(-Z))
        dZ = dA * A * (1 - A)
    else:
        raise ValueError(f"Unknown activation: {activation}")
    return dZ

def level2_forward_pass(X, params, activation):
    """Run a configurable MLP forward pass for the Level 2 builder."""
    weights, biases = level2_deserialize_params(params)
    activations = [X.T]
    pre_activations = []
    current_activation = X.T

    for weight, bias in zip(weights[:-1], biases[:-1]):
        Z = weight @ current_activation + bias
        pre_activations.append(Z)
        current_activation = activation_forward(Z, activation)
        activations.append(current_activation)

    Z_out = weights[-1] @ current_activation + biases[-1]
    A_out = 1.0 / (1.0 + np.exp(-Z_out))
    pre_activations.append(Z_out)
    activations.append(A_out)

    return A_out, {
        'activations': activations,
        'pre_activations': pre_activations,
    }


def level2_evaluate_metrics(X, y, params, activation, l2=0.0):
    predictions, _ = level2_forward_pass(X, params, activation)
    y_row = y.reshape(1, -1)
    eps = 1e-8
    loss = -np.mean(y_row * np.log(predictions + eps) + (1 - y_row) * np.log(1 - predictions + eps))

    if l2 > 0:
        weights, _ = level2_deserialize_params(params)
        loss += 0.5 * l2 * sum(np.sum(weight * weight) for weight in weights)

    accuracy = np.mean((predictions >= 0.5).astype(np.int32) == y_row)
    return {
        'loss': float(loss),
        'accuracy': float(accuracy),
        'epoch': int(params.get('epoch', 0)),
        'parameter_count': level2_parameter_count(params),
    }


def level2_set_baseline_history(X, y, params, activation, l2=0.0):
    metrics = level2_evaluate_metrics(X, y, params, activation, l2=l2)
    params['epoch'] = 0
    params['history'] = {
        'epochs': [0],
        'loss': [metrics['loss']],
        'accuracy': [metrics['accuracy']],
    }
    return params


def train_level2_model(X, y, params, activation='tanh', epochs=80, lr=0.08, l2=1e-4):
    """Train the Level 2 configurable MLP with full-batch gradient descent."""
    weights, biases = level2_deserialize_params(params)
    y_row = y.reshape(1, -1)
    sample_count = X.shape[0]

    history = params.get('history', {'epochs': [0], 'loss': [], 'accuracy': []})
    epochs_history = list(history.get('epochs', [0]))
    loss_history = list(history.get('loss', []))
    accuracy_history = list(history.get('accuracy', []))

    for _ in range(epochs):
        predictions, cache = level2_forward_pass(
            X,
            {'weights': [weight.tolist() for weight in weights], 'biases': [bias.tolist() for bias in biases]},
            activation
        )
        dZ = predictions - y_row
        gradients_w = [None] * len(weights)
        gradients_b = [None] * len(biases)

        for layer_index in reversed(range(len(weights))):
            prev_activation = cache['activations'][layer_index]
            gradients_w[layer_index] = (dZ @ prev_activation.T) / sample_count
            gradients_b[layer_index] = np.mean(dZ, axis=1, keepdims=True)

            if l2 > 0:
                gradients_w[layer_index] += l2 * weights[layer_index]

            if layer_index > 0:
                dA_prev = weights[layer_index].T @ dZ
                dZ = activation_backward(
                    dA_prev,
                    cache['pre_activations'][layer_index - 1],
                    activation
                )

        for layer_index in range(len(weights)):
            weights[layer_index] -= lr * gradients_w[layer_index]
            biases[layer_index] -= lr * gradients_b[layer_index]

        params['epoch'] = int(params.get('epoch', 0)) + 1
        serialized_params = {
            'weights': [weight.tolist() for weight in weights],
            'biases': [bias.tolist() for bias in biases],
            'epoch': params['epoch'],
        }
        metrics = level2_evaluate_metrics(X, y, serialized_params, activation, l2=l2)
        epochs_history.append(params['epoch'])
        loss_history.append(metrics['loss'])
        accuracy_history.append(metrics['accuracy'])

    params = level2_serialize_params(weights, biases, params)
    params['history'] = {
        'epochs': epochs_history,
        'loss': loss_history,
        'accuracy': accuracy_history,
    }
    return params

def make_decision_boundary_figure(X, y, params, activation, grid_step=0.03):
    # Bounds
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Forward pass on grid
    A2_grid, _ = level2_forward_pass(grid_points, params, activation)
    Z = A2_grid.reshape(xx.shape)  # predicted probability

    contour = go.Contour(
        x=xx[0, :],
        y=yy[:, 0],
        z=Z,
        showscale=False,
        contours=dict(showlines=False),
        colorscale='RdBu',
        opacity=0.6
    )

    scatter = go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(
            color=y,
            colorscale='Viridis',
            line=dict(width=1, color='black'),
            size=7
        ),
        name='Data'
    )

    fig = go.Figure(data=[contour, scatter])
    fig.update_layout(
        title="Decision boundary in input space",
        xaxis_title="x₁",
        yaxis_title="x₂",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def make_activation_figure(activation):
    z = np.linspace(-5, 5, 400)
    if activation == 'relu':
        a = np.maximum(0, z)
        title = "ReLU activation ρ(z) = max(0, z)"
    elif activation == 'tanh':
        a = np.tanh(z)
        title = "Tanh activation ρ(z) = tanh(z)"
    elif activation == 'sigmoid':
        a = 1.0 / (1.0 + np.exp(-z))
        title = "Sigmoid activation ρ(z) = 1 / (1 + e^{-z})"
    else:
        a = z
        title = f"Unknown activation: {activation}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=a, mode='lines'))
    fig.update_layout(
        title=title,
        xaxis_title="z",
        yaxis_title="ρ(z)",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def make_network_diagram_figure(input_dim=2, hidden_layers=None, output_dim=1,
                                params=None, activation='tanh'):
    if hidden_layers is None:
        hidden_layers = [6, 6]

    layer_sizes = [input_dim] + list(hidden_layers) + [output_dim]
    layer_x_positions = np.linspace(0, 1, len(layer_sizes))
    weights, biases = level2_deserialize_params(params) if params else (None, None)

    nodes_x = []
    nodes_y = []
    labels = []
    hover_text = []
    colors = []
    edge_traces = []
    layer_coordinates = []
    annotations = []

    for layer_index, layer_size in enumerate(layer_sizes):
        y_positions = np.linspace(0, 1, layer_size)
        layer_coordinates.append(y_positions)
        if layer_index == 0:
            layer_name = 'Input'
            color = '#93c5fd'
        elif layer_index == len(layer_sizes) - 1:
            layer_name = 'Output'
            color = '#fca5a5'
        else:
            layer_name = f'Hidden {layer_index}'
            color = '#86efac'

        annotations.append(dict(
            x=layer_x_positions[layer_index],
            y=1.1,
            text=f"{layer_name}<br>{layer_size} neuron(s)",
            showarrow=False,
            font=dict(size=11)
        ))

        for node_index, y_position in enumerate(y_positions):
            nodes_x.append(layer_x_positions[layer_index])
            nodes_y.append(y_position)
            colors.append(color)

            if layer_index == 0:
                labels.append(f"x{node_index + 1}")
                hover_text.append(f"Input feature x{node_index + 1}")
            elif layer_index == len(layer_sizes) - 1:
                labels.append('ŷ')
                bias_value = biases[layer_index - 1][node_index, 0] if biases is not None else 0.0
                hover_text.append(
                    f"Output node<br>bias={bias_value:+.3f}<br>ŷ = σ(z)"
                )
            else:
                labels.append(f"h{layer_index}.{node_index + 1}")
                bias_value = biases[layer_index - 1][node_index, 0] if biases is not None else 0.0
                hover_text.append(
                    f"Layer {layer_index} neuron {node_index + 1}<br>bias={bias_value:+.3f}<br>a = {activation}(z)"
                )

    for layer_index in range(len(layer_sizes) - 1):
        source_x = layer_x_positions[layer_index]
        target_x = layer_x_positions[layer_index + 1]
        source_y_positions = layer_coordinates[layer_index]
        target_y_positions = layer_coordinates[layer_index + 1]
        current_weights = weights[layer_index] if weights is not None else None

        for source_index, source_y in enumerate(source_y_positions):
            for target_index, target_y in enumerate(target_y_positions):
                if current_weights is not None:
                    weight_value = current_weights[target_index, source_index]
                    edge_color = 'steelblue' if weight_value >= 0 else 'crimson'
                    edge_width = max(0.5, min(4.5, abs(weight_value) * 2.5))
                    edge_hover = (
                        f"Layer {layer_index + 1} weight[{target_index + 1},{source_index + 1}] = "
                        f"{weight_value:+.3f}"
                    )
                else:
                    edge_color = 'grey'
                    edge_width = 1
                    edge_hover = 'Weight'

                edge_traces.append(go.Scatter(
                    x=[source_x, (source_x + target_x) / 2, target_x],
                    y=[source_y, (source_y + target_y) / 2, target_y],
                    mode='lines',
                    line=dict(color=edge_color, width=edge_width),
                    hovertext=[None, edge_hover, None],
                    hoverinfo='text',
                    showlegend=False
                ))

    node_trace = go.Scatter(
        x=nodes_x,
        y=nodes_y,
        mode='markers+text',
        text=labels,
        textposition='middle right',
        hovertext=hover_text,
        hoverinfo='text',
        marker=dict(size=16, color=colors, line=dict(width=1, color='black'))
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title='User-built network architecture',
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=0, r=0, t=70, b=0),
        showlegend=False,
        annotations=annotations,
    )
    return fig


def make_level2_training_curves_figure(history):
    epochs = history.get('epochs', [])
    losses = history.get('loss', [])
    accuracies = history.get('accuracy', [])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=losses,
        mode='lines+markers',
        name='Loss',
        line=dict(color='#dc2626', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=epochs,
        y=accuracies,
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='#0f766e', width=3),
        yaxis='y2'
    ))
    fig.update_layout(
        title='Optimisation behaviour across training runs',
        xaxis_title='Epoch',
        yaxis=dict(title='Loss'),
        yaxis2=dict(title='Accuracy', overlaying='y', side='right', range=[0, 1]),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def make_level2_metrics_cards(metrics):
    card_style = {
        'backgroundColor': 'white',
        'borderRadius': '14px',
        'padding': '14px 16px',
        'boxShadow': '0 2px 8px rgba(15, 23, 42, 0.08)',
        'border': '1px solid #e5e7eb'
    }
    label_style = {'fontSize': '12px', 'textTransform': 'uppercase', 'color': '#64748b', 'letterSpacing': '0.08em'}
    value_style = {'fontSize': '24px', 'fontWeight': '700', 'marginTop': '6px'}
    subtitle_style = {'fontSize': '12px', 'color': '#64748b', 'marginTop': '4px'}

    return [
        html.Div([
            html.Div('Accuracy', style=label_style),
            html.Div(f"{metrics['accuracy'] * 100:.1f}%", style=value_style),
            html.Div('Classification performance on the selected toy dataset.', style=subtitle_style),
        ], style=card_style),
        html.Div([
            html.Div('Loss', style=label_style),
            html.Div(f"{metrics['loss']:.4f}", style=value_style),
            html.Div('Binary cross-entropy after the current training run.', style=subtitle_style),
        ], style=card_style),
        html.Div([
            html.Div('Parameters', style=label_style),
            html.Div(str(metrics['parameter_count']), style=value_style),
            html.Div('Trainable weights and biases in the current builder architecture.', style=subtitle_style),
        ], style=card_style),
        html.Div([
            html.Div('Epoch', style=label_style),
            html.Div(str(metrics['epoch']), style=value_style),
            html.Div('Accumulated full-batch training epochs for this run.', style=subtitle_style),
        ], style=card_style),
    ]


def make_level2_summary_panel(params, activation):
    meta = params.get('meta', {})
    hidden_layers = meta.get('hidden_layer_sizes', [6, 6])
    layer_sizes = [2] + list(hidden_layers) + [1]
    weights, biases = level2_deserialize_params(params)
    layer_items = []

    for layer_index, (weight, bias) in enumerate(zip(weights, biases), start=1):
        if layer_index < len(weights):
            layer_label = f"Hidden {layer_index}"
            transform = activation
        else:
            layer_label = 'Output'
            transform = 'sigmoid'

        layer_items.append(
            html.Li(
                f"{layer_label}: W{layer_index} shape {weight.shape}, b{layer_index} shape {bias.shape}, activation={transform}",
                style={'fontFamily': 'monospace', 'fontSize': '11px'}
            )
        )

    return html.Div([
        html.Div(
            f"Architecture: {' → '.join(str(size) for size in layer_sizes)}",
            style={'fontWeight': '700', 'marginBottom': '10px'}
        ),
        html.P(
            "Forward map: h⁽ˡ⁾ = ρ(W⁽ˡ⁾h⁽ˡ⁻¹⁾ + b⁽ˡ⁾), with a sigmoid output layer for binary classification.",
            style={'fontSize': '13px', 'marginBottom': '10px'}
        ),
        html.P(
            f"Hidden activation: {activation} | Total parameters: {level2_parameter_count(params)}",
            style={'fontSize': '13px', 'color': '#475569'}
        ),
        html.Ul(layer_items, style={'paddingLeft': '18px', 'marginBottom': '10px'}),
        html.Div(
            "Tip: add layers or neurons to increase expressivity, then compare whether the extra capacity actually improves the learned boundary and training curves.",
            style={'fontSize': '12px', 'color': '#64748b'}
        )
    ])


def make_level2_comparison_panel(compare_store, current_metrics):
    if not compare_store:
        return html.Div([
            html.H4('Comparison Run', style={'marginTop': '0'}),
            html.P(
                'Save a baseline with Compare Run, then change the architecture or train again to inspect the difference in accuracy, loss, and model size.',
                style={'fontSize': '13px', 'color': '#64748b', 'marginBottom': '0'}
            ),
        ])

    saved_metrics = compare_store['metrics']
    saved_meta = compare_store['meta']
    accuracy_delta = current_metrics['accuracy'] - saved_metrics['accuracy']
    loss_delta = current_metrics['loss'] - saved_metrics['loss']
    parameter_delta = current_metrics['parameter_count'] - saved_metrics['parameter_count']

    return html.Div([
        html.H4('Comparison Run', style={'marginTop': '0'}),
        html.P(
            f"Saved baseline: {saved_meta['dataset']} | {' → '.join(str(size) for size in saved_meta['layer_sizes'])} | {saved_meta['activation']}",
            style={'fontSize': '13px', 'marginBottom': '10px'}
        ),
        html.Div(f"Accuracy delta: {accuracy_delta * 100:+.1f}%", style={'fontWeight': '600', 'marginBottom': '6px'}),
        html.Div(f"Loss delta: {loss_delta:+.4f}", style={'fontWeight': '600', 'marginBottom': '6px'}),
        html.Div(f"Parameter delta: {parameter_delta:+d}", style={'fontWeight': '600', 'marginBottom': '10px'}),
        html.P(
            'Use this panel to decide whether added capacity improved general behaviour or only made optimisation heavier.',
            style={'fontSize': '12px', 'color': '#64748b', 'marginBottom': '0'}
        ),
    ])

def register_level2_callbacks(app):
    @app.callback(
    Output('level2-params-store', 'data'),
    Output('level2-compare-store', 'data'),
    Input('level2-train-btn', 'n_clicks'),
    Input('level2-reset-btn', 'n_clicks'),
    Input('level2-compare-btn', 'n_clicks'),
    Input('level2-hidden-layers-slider', 'value'),
    Input('level2-neurons-slider', 'value'),
    Input('level2-activation-dropdown', 'value'),
    Input('level2-dataset-dropdown', 'value'),
    State('level2-params-store', 'data'),
    State('level2-compare-store', 'data'),
    )
    
    def update_level2_params(n_train, n_reset, n_compare, hidden_layers, neurons_per_layer,
                         activation, dataset, params, compare_store):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    hidden_layer_sizes = [neurons_per_layer] * hidden_layers
    meta = {
        'hidden_layers': hidden_layers,
        'neurons_per_layer': neurons_per_layer,
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'dataset': dataset,
        'layer_sizes': [2] + hidden_layer_sizes + [1],
    }
    rebuild_triggers = {
        None,
        'level2-reset-btn',
        'level2-hidden-layers-slider',
        'level2-neurons-slider',
        'level2-activation-dropdown',
        'level2-dataset-dropdown',
    }

    if params is None or trigger in rebuild_triggers:
        params = init_level2_mlp(input_dim=2, hidden_layers=hidden_layer_sizes, output_dim=1)
        params['meta'] = meta
        X, y = load_toy_dataset(dataset)
        params = level2_set_baseline_history(X, y, params, activation, l2=1e-4)

    if trigger == 'level2-train-btn':
        params['meta'] = meta
        X, y = load_toy_dataset(dataset)
        params = train_level2_model(X, y, params, activation=activation, epochs=80, lr=0.08, l2=1e-4)

    if trigger == 'level2-compare-btn':
        params['meta'] = meta
        X, y = load_toy_dataset(dataset)
        metrics = level2_evaluate_metrics(X, y, params, activation, l2=1e-4)
        compare_store = {
            'meta': meta,
            'metrics': metrics,
        }

    params['meta'] = meta
    return params, compare_store
    
    @app.callback(
    Output('level2-decision-boundary-graph', 'figure'),
    Output('level2-activation-graph', 'figure'),
    Output('level2-network-diagram-graph', 'figure'),
    Output('level2-math-explanation', 'children'),
    Output('level2-metrics-row', 'children'),
    Output('level2-comparison-panel', 'children'),
    Output('level2-training-curves-graph', 'figure'),
    Input('level2-params-store', 'data'),
    Input('level2-compare-store', 'data')
)
def update_level2_views(params, compare_store):
    if params is None:
        raise dash.exceptions.PreventUpdate

    meta = params.get('meta', {})
    hidden_layer_sizes = meta.get('hidden_layer_sizes', [6, 6])
    activation = meta.get('activation', 'tanh')
    dataset = meta.get('dataset', 'moons')

    X, y = load_toy_dataset(dataset)
    metrics = level2_evaluate_metrics(X, y, params, activation, l2=1e-4)

    fig_boundary = make_decision_boundary_figure(X, y, params, activation)
    fig_boundary.update_layout(
        title=(
            f"{dataset.title()} dataset | Architecture {' → '.join(str(size) for size in meta.get('layer_sizes', [2] + hidden_layer_sizes + [1]))}"
        )
    )

    fig_activation = make_activation_figure(activation)
    fig_network = make_network_diagram_figure(
        input_dim=2,
        hidden_layers=hidden_layer_sizes,
        output_dim=1,
        params=params, activation=activation
    )
    explanation = make_level2_summary_panel(params, activation)
    metric_cards = make_level2_metrics_cards(metrics)
    comparison_panel = make_level2_comparison_panel(compare_store, metrics)
    curves_figure = make_level2_training_curves_figure(params.get('history', {}))

    return (
        fig_boundary,
        fig_activation,
        fig_network,
        explanation,
        metric_cards,
        comparison_panel,
        curves_figure,
    )

#endregion
