import numpy as np
import plotly.graph_objects as go
from dash import html
from sklearn.datasets import make_circles, make_classification, make_moons


def load_toy_dataset(name, n_samples=300, noise=0.2, random_state=0):
    if name == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif name == 'circles':
        X, y = make_circles(
            n_samples=n_samples,
            factor=0.5,
            noise=noise,
            random_state=random_state,
        )
    elif name == 'linear':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=1.5,
            random_state=random_state,
        )
    else:
        raise ValueError(f'Unknown dataset: {name}')

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
        weights.append(rng.normal(0.0, 1.0 / np.sqrt(fan_in), size=(fan_out, fan_in)))
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


def activation_forward(z_values, activation):
    if activation == 'relu':
        return np.maximum(0, z_values)
    if activation == 'tanh':
        return np.tanh(z_values)
    if activation == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-z_values))
    raise ValueError(f'Unknown activation: {activation}')


def activation_backward(d_activation, z_values, activation):
    if activation == 'relu':
        return d_activation * (z_values > 0)
    if activation == 'tanh':
        activations = np.tanh(z_values)
        return d_activation * (1 - activations ** 2)
    if activation == 'sigmoid':
        activations = 1.0 / (1.0 + np.exp(-z_values))
        return d_activation * activations * (1 - activations)
    raise ValueError(f'Unknown activation: {activation}')


def level2_forward_pass(X, params, activation):
    weights, biases = level2_deserialize_params(params)
    activations = [X.T]
    pre_activations = []
    current_activation = X.T

    for weight, bias in zip(weights[:-1], biases[:-1]):
        z_values = weight @ current_activation + bias
        pre_activations.append(z_values)
        current_activation = activation_forward(z_values, activation)
        activations.append(current_activation)

    z_output = weights[-1] @ current_activation + biases[-1]
    output = 1.0 / (1.0 + np.exp(-z_output))
    pre_activations.append(z_output)
    activations.append(output)

    return output, {
        'activations': activations,
        'pre_activations': pre_activations,
    }


def level2_evaluate_metrics(X, y, params, activation, l2=0.0):
    predictions, _ = level2_forward_pass(X, params, activation)
    y_row = y.reshape(1, -1)
    eps = 1e-8
    loss = -np.mean(
        y_row * np.log(predictions + eps)
        + (1 - y_row) * np.log(1 - predictions + eps)
    )

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
            {
                'weights': [weight.tolist() for weight in weights],
                'biases': [bias.tolist() for bias in biases],
            },
            activation,
        )
        d_z = predictions - y_row
        gradients_w = [None] * len(weights)
        gradients_b = [None] * len(biases)

        for layer_index in reversed(range(len(weights))):
            prev_activation = cache['activations'][layer_index]
            gradients_w[layer_index] = (d_z @ prev_activation.T) / sample_count
            gradients_b[layer_index] = np.mean(d_z, axis=1, keepdims=True)

            if l2 > 0:
                gradients_w[layer_index] += l2 * weights[layer_index]

            if layer_index > 0:
                d_activation = weights[layer_index].T @ d_z
                d_z = activation_backward(
                    d_activation,
                    cache['pre_activations'][layer_index - 1],
                    activation,
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
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    grid_predictions, _ = level2_forward_pass(grid_points, params, activation)
    z_values = grid_predictions.reshape(xx.shape)

    contour = go.Contour(
        x=xx[0, :],
        y=yy[:, 0],
        z=z_values,
        showscale=False,
        contours=dict(showlines=False),
        colorscale='RdBu',
        opacity=0.6,
    )
    scatter = go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(
            color=y,
            colorscale='Viridis',
            line=dict(width=1, color='black'),
            size=7,
        ),
        name='Data',
    )

    fig = go.Figure(data=[contour, scatter])
    fig.update_layout(
        title='Decision boundary in input space',
        xaxis_title='x1',
        yaxis_title='x2',
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def make_activation_figure(activation):
    z_values = np.linspace(-5, 5, 400)
    if activation == 'relu':
        activations = np.maximum(0, z_values)
        title = 'ReLU activation rho(z) = max(0, z)'
    elif activation == 'tanh':
        activations = np.tanh(z_values)
        title = 'Tanh activation rho(z) = tanh(z)'
    elif activation == 'sigmoid':
        activations = 1.0 / (1.0 + np.exp(-z_values))
        title = 'Sigmoid activation rho(z) = 1 / (1 + e^{-z})'
    else:
        activations = z_values
        title = f'Unknown activation: {activation}'

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z_values, y=activations, mode='lines'))
    fig.update_layout(
        title=title,
        xaxis_title='z',
        yaxis_title='rho(z)',
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def make_network_diagram_figure(input_dim=2, hidden_layers=None, output_dim=1, params=None, activation='tanh'):
    if hidden_layers is None:
        hidden_layers = [6, 6]

    layer_sizes = [input_dim] + list(hidden_layers) + [output_dim]
    layer_x_positions = np.linspace(0, 1, len(layer_sizes))
    weights, biases = level2_deserialize_params(params) if params else (None, None)

    nodes_x = []
    nodes_y = []
    labels = []
    hover_text = []
    colours = []
    edge_traces = []
    layer_coordinates = []
    annotations = []

    for layer_index, layer_size in enumerate(layer_sizes):
        y_positions = np.linspace(0, 1, layer_size)
        layer_coordinates.append(y_positions)
        if layer_index == 0:
            layer_name = 'Input'
            colour = '#93c5fd'
        elif layer_index == len(layer_sizes) - 1:
            layer_name = 'Output'
            colour = '#fca5a5'
        else:
            layer_name = f'Hidden {layer_index}'
            colour = '#86efac'

        annotations.append(
            dict(
                x=layer_x_positions[layer_index],
                y=1.1,
                text=f'{layer_name}<br>{layer_size} neuron(s)',
                showarrow=False,
                font=dict(size=11),
            )
        )

        for node_index, y_position in enumerate(y_positions):
            nodes_x.append(layer_x_positions[layer_index])
            nodes_y.append(y_position)
            colours.append(colour)

            if layer_index == 0:
                labels.append(f'x{node_index + 1}')
                hover_text.append(f'Input feature x{node_index + 1}')
            elif layer_index == len(layer_sizes) - 1:
                labels.append('yhat')
                bias_value = biases[layer_index - 1][node_index, 0] if biases is not None else 0.0
                hover_text.append(f'Output node<br>bias={bias_value:+.3f}<br>yhat = sigmoid(z)')
            else:
                labels.append(f'h{layer_index}.{node_index + 1}')
                bias_value = biases[layer_index - 1][node_index, 0] if biases is not None else 0.0
                hover_text.append(
                    f'Layer {layer_index} neuron {node_index + 1}<br>bias={bias_value:+.3f}<br>a = {activation}(z)'
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
                    edge_colour = 'steelblue' if weight_value >= 0 else 'crimson'
                    edge_width = max(0.5, min(4.5, abs(weight_value) * 2.5))
                    edge_hover = (
                        f'Layer {layer_index + 1} weight[{target_index + 1},{source_index + 1}] = '
                        f'{weight_value:+.3f}'
                    )
                else:
                    edge_colour = 'grey'
                    edge_width = 1
                    edge_hover = 'Weight'

                edge_traces.append(
                    go.Scatter(
                        x=[source_x, (source_x + target_x) / 2, target_x],
                        y=[source_y, (source_y + target_y) / 2, target_y],
                        mode='lines',
                        line=dict(color=edge_colour, width=edge_width),
                        hovertext=[None, edge_hover, None],
                        hoverinfo='text',
                        showlegend=False,
                    )
                )

    node_trace = go.Scatter(
        x=nodes_x,
        y=nodes_y,
        mode='markers+text',
        text=labels,
        textposition='middle right',
        hovertext=hover_text,
        hoverinfo='text',
        marker=dict(size=16, color=colours, line=dict(width=1, color='black')),
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
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=losses,
            mode='lines+markers',
            name='Loss',
            line=dict(color='#dc2626', width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=accuracies,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#0f766e', width=3),
            yaxis='y2',
        )
    )
    fig.update_layout(
        title='Optimisation behaviour across training runs',
        xaxis_title='Epoch',
        yaxis=dict(title='Loss'),
        yaxis2=dict(title='Accuracy', overlaying='y', side='right', range=[0, 1]),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


def make_level2_metrics_cards(metrics):
    card_style = {
        'backgroundColor': 'white',
        'borderRadius': '14px',
        'padding': '14px 16px',
        'boxShadow': '0 2px 8px rgba(15, 23, 42, 0.08)',
        'border': '1px solid #e5e7eb',
    }
    label_style = {
        'fontSize': '12px',
        'textTransform': 'uppercase',
        'color': '#64748b',
        'letterSpacing': '0.08em',
    }
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
            layer_label = f'Hidden {layer_index}'
            transform = activation
        else:
            layer_label = 'Output'
            transform = 'sigmoid'

        layer_items.append(
            html.Li(
                f'{layer_label}: W{layer_index} shape {weight.shape}, b{layer_index} shape {bias.shape}, activation={transform}',
                style={'fontFamily': 'monospace', 'fontSize': '11px'},
            )
        )

    return html.Div([
        html.Div(
            f"Architecture: {' -> '.join(str(size) for size in layer_sizes)}",
            style={'fontWeight': '700', 'marginBottom': '10px'},
        ),
        html.P(
            'Forward map: h(l) = rho(W(l)h(l-1) + b(l)), with a sigmoid output layer for binary classification.',
            style={'fontSize': '13px', 'marginBottom': '10px'},
        ),
        html.P(
            f'Hidden activation: {activation} | Total parameters: {level2_parameter_count(params)}',
            style={'fontSize': '13px', 'color': '#475569'},
        ),
        html.Ul(layer_items, style={'paddingLeft': '18px', 'marginBottom': '10px'}),
        html.Div(
            'Tip: add layers or neurons to increase expressivity, then compare whether the extra capacity actually improves the learned boundary and training curves.',
            style={'fontSize': '12px', 'color': '#64748b'},
        ),
    ])


def make_level2_comparison_panel(compare_store, current_metrics):
    if not compare_store:
        return html.Div([
            html.H4('Comparison Run', style={'marginTop': '0'}),
            html.P(
                'Save a baseline with Compare Run, then change the architecture or train again to inspect the difference in accuracy, loss, and model size.',
                style={'fontSize': '13px', 'color': '#64748b', 'marginBottom': '0'},
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
            f"Saved baseline: {saved_meta['dataset']} | {' -> '.join(str(size) for size in saved_meta['layer_sizes'])} | {saved_meta['activation']}",
            style={'fontSize': '13px', 'marginBottom': '10px'},
        ),
        html.Div(f'Accuracy delta: {accuracy_delta * 100:+.1f}%', style={'fontWeight': '600', 'marginBottom': '6px'}),
        html.Div(f'Loss delta: {loss_delta:+.4f}', style={'fontWeight': '600', 'marginBottom': '6px'}),
        html.Div(f'Parameter delta: {parameter_delta:+d}', style={'fontWeight': '600', 'marginBottom': '10px'}),
        html.P(
            'Use this panel to decide whether added capacity improved general behaviour or only made optimisation heavier.',
            style={'fontSize': '12px', 'color': '#64748b', 'marginBottom': '0'},
        ),
    ])