import numpy as np
import plotly.graph_objects as go
from dash import html
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split


FEATURE_NAMES = [
    'x1',
    'x2',
    'x1 * x2',
    'x1^2',
    'x2^2',
    'sin(pi * x1)',
    'sin(pi * x2)',
    'x1 - x2',
]


def load_toy_dataset(name, n_samples=320, noise=0.2, random_state=0):
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


def expand_level2_features(X_raw, input_dim, stats=None, return_stats=False):
    x1 = X_raw[:, 0]
    x2 = X_raw[:, 1]
    feature_bank = np.column_stack([
        x1,
        x2,
        x1 * x2,
        x1 ** 2,
        x2 ** 2,
        np.sin(np.pi * x1),
        np.sin(np.pi * x2),
        x1 - x2,
    ]).astype(np.float64)

    safe_input_dim = max(2, min(int(input_dim), feature_bank.shape[1]))
    selected_features = feature_bank[:, :safe_input_dim]

    if stats is None:
        mean = selected_features.mean(axis=0, keepdims=True)
        std = selected_features.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        stats = {'mean': mean, 'std': std}

    normalized = (selected_features - stats['mean']) / stats['std']

    if return_stats:
        return normalized.astype(np.float32), FEATURE_NAMES[:safe_input_dim], stats

    return normalized.astype(np.float32), FEATURE_NAMES[:safe_input_dim]


def build_level2_dataset(name, input_dim=2, test_size=0.25, random_state=7):
    X_raw, y = load_toy_dataset(name, random_state=random_state)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train, feature_names, stats = expand_level2_features(
        X_train_raw,
        input_dim,
        return_stats=True,
    )
    X_test, _ = expand_level2_features(X_test_raw, input_dim, stats=stats)
    X_all, _ = expand_level2_features(X_raw, input_dim, stats=stats)

    return {
        'dataset_name': name,
        'feature_names': feature_names,
        'transform_stats': stats,
        'X_raw': X_raw,
        'y': y,
        'X_all': X_all,
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train.astype(np.int32),
        'y_test': y_test.astype(np.int32),
    }


def init_level2_mlp(input_dim=2, hidden_layers=None, output_dim=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if hidden_layers is None:
        hidden_layers = [6, 6]

    layer_sizes = [input_dim] + list(hidden_layers) + [output_dim]
    weights = []
    biases = []

    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        weights.append(rng.normal(0.0, 1.0 / np.sqrt(max(fan_in, 1)), size=(fan_out, fan_in)))
        biases.append(np.zeros((fan_out, 1)))

    return {
        'weights': [weight.tolist() for weight in weights],
        'biases': [bias.tolist() for bias in biases],
        'epoch': 0,
        'history': {
            'epochs': [0],
            'train_loss': [],
            'test_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
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


def _copy_serialized_weights(weights):
    return [[list(row) for row in layer] for layer in weights]


def _activation_to_colour(value):
    magnitude = min(1.0, abs(float(value)))
    neutral = np.array([241.0, 245.0, 249.0])
    positive = np.array([249.0, 115.0, 22.0])
    negative = np.array([37.0, 99.0, 235.0])
    anchor = positive if value >= 0 else negative
    rgb = ((1.0 - magnitude) * neutral) + (magnitude * anchor)
    return f'rgb({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])})'


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


def output_forward(z_values):
    if z_values.shape[0] == 1:
        return 1.0 / (1.0 + np.exp(-z_values))

    shifted = z_values - np.max(z_values, axis=0, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)


def level2_make_targets(y, output_dim):
    if output_dim == 1:
        return y.reshape(1, -1).astype(np.float64)

    classes = np.eye(output_dim, dtype=np.float64)
    return classes[y].T


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
    output = output_forward(z_output)
    pre_activations.append(z_output)
    activations.append(output)

    return output, {
        'activations': activations,
        'pre_activations': pre_activations,
    }


def make_level2_activation_snapshot(dataset_bundle, params, activation, probe_index=0):
    X_train = dataset_bundle['X_train']
    X_train_raw = dataset_bundle['X_train_raw']
    y_train = dataset_bundle['y_train']

    if X_train.shape[0] == 0:
        return None

    safe_probe_index = int(probe_index) % X_train.shape[0]
    probe_features = X_train[safe_probe_index:safe_probe_index + 1]
    probe_raw = X_train_raw[safe_probe_index]
    probe_label = int(y_train[safe_probe_index])
    predictions, cache = level2_forward_pass(probe_features, params, activation)
    output_vector = predictions[:, 0]

    if output_vector.shape[0] == 1:
        positive_probability = float(output_vector[0])
        predicted_label = int(positive_probability >= 0.5)
        confidence = positive_probability if predicted_label == 1 else 1.0 - positive_probability
    else:
        predicted_label = int(np.argmax(output_vector))
        confidence = float(output_vector[predicted_label])

    layer_activations = [
        activation_values[:, 0].astype(float).tolist()
        for activation_values in cache['activations']
    ]

    return {
        'probe_index': safe_probe_index,
        'probe_raw': probe_raw.astype(float).tolist(),
        'probe_label': probe_label,
        'predicted_label': predicted_label,
        'confidence': confidence,
        'prediction_vector': output_vector.astype(float).tolist(),
        'layer_activations': layer_activations,
    }


def level2_evaluate_split(X, y, params, activation, l2=0.0):
    predictions, _ = level2_forward_pass(X, params, activation)
    output_dim = predictions.shape[0]
    targets = level2_make_targets(y, output_dim)
    eps = 1e-8

    if output_dim == 1:
        loss = -np.mean(
            targets * np.log(predictions + eps)
            + (1 - targets) * np.log(1 - predictions + eps)
        )
        predicted_labels = (predictions >= 0.5).astype(np.int32).ravel()
    else:
        loss = -np.mean(np.sum(targets * np.log(predictions + eps), axis=0))
        predicted_labels = np.argmax(predictions, axis=0).astype(np.int32)

    if l2 > 0:
        weights, _ = level2_deserialize_params(params)
        loss += 0.5 * l2 * sum(np.sum(weight * weight) for weight in weights)

    accuracy = np.mean(predicted_labels == y)
    return float(loss), float(accuracy)


def level2_evaluate_metrics(dataset_bundle, params, activation, l2=0.0):
    train_loss, train_accuracy = level2_evaluate_split(
        dataset_bundle['X_train'],
        dataset_bundle['y_train'],
        params,
        activation,
        l2=l2,
    )
    test_loss, test_accuracy = level2_evaluate_split(
        dataset_bundle['X_test'],
        dataset_bundle['y_test'],
        params,
        activation,
        l2=l2,
    )

    return {
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'epoch': int(params.get('epoch', 0)),
        'parameter_count': level2_parameter_count(params),
    }


def level2_set_baseline_history(dataset_bundle, params, activation, l2=0.0):
    metrics = level2_evaluate_metrics(dataset_bundle, params, activation, l2=l2)
    params['epoch'] = 0
    params['history'] = {
        'epochs': [0],
        'train_loss': [metrics['train_loss']],
        'test_loss': [metrics['test_loss']],
        'train_accuracy': [metrics['train_accuracy']],
        'test_accuracy': [metrics['test_accuracy']],
    }
    return params


def train_level2_model(dataset_bundle, params, activation='tanh', epochs=1, lr=0.08, l2=1e-4):
    weights, biases = level2_deserialize_params(params)
    X_train = dataset_bundle['X_train']
    y_train = dataset_bundle['y_train']
    targets = level2_make_targets(y_train, weights[-1].shape[0])
    sample_count = X_train.shape[0]

    history = params.get(
        'history',
        {
            'epochs': [0],
            'train_loss': [],
            'test_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
        },
    )
    epochs_history = list(history.get('epochs', [0]))
    train_loss_history = list(history.get('train_loss', []))
    test_loss_history = list(history.get('test_loss', []))
    train_accuracy_history = list(history.get('train_accuracy', []))
    test_accuracy_history = list(history.get('test_accuracy', []))

    for _ in range(epochs):
        predictions, cache = level2_forward_pass(
            X_train,
            {
                'weights': [weight.tolist() for weight in weights],
                'biases': [bias.tolist() for bias in biases],
            },
            activation,
        )
        d_z = predictions - targets
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
        metrics = level2_evaluate_metrics(dataset_bundle, serialized_params, activation, l2=l2)
        epochs_history.append(params['epoch'])
        train_loss_history.append(metrics['train_loss'])
        test_loss_history.append(metrics['test_loss'])
        train_accuracy_history.append(metrics['train_accuracy'])
        test_accuracy_history.append(metrics['test_accuracy'])

    params = level2_serialize_params(weights, biases, params)
    params['history'] = {
        'epochs': epochs_history,
        'train_loss': train_loss_history,
        'test_loss': test_loss_history,
        'train_accuracy': train_accuracy_history,
        'test_accuracy': test_accuracy_history,
    }
    return params


def make_decision_boundary_figure(dataset_bundle, params, activation, grid_step=0.03):
    X_raw = dataset_bundle['X_raw']
    y = dataset_bundle['y']
    input_dim = len(dataset_bundle['feature_names'])
    stats = dataset_bundle['transform_stats']

    x_min, x_max = X_raw[:, 0].min() - 0.5, X_raw[:, 0].max() + 0.5
    y_min, y_max = X_raw[:, 1].min() - 0.5, X_raw[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step),
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_features, _ = expand_level2_features(grid_points, input_dim, stats=stats)

    grid_predictions, _ = level2_forward_pass(grid_features, params, activation)
    if grid_predictions.shape[0] == 1:
        z_values = grid_predictions.reshape(xx.shape)
    else:
        z_values = grid_predictions[1].reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=xx[0, :],
            y=yy[:, 0],
            z=z_values,
            showscale=False,
            contours=dict(showlines=False),
            colorscale='RdBu',
            opacity=0.6,
            hoverinfo='skip',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataset_bundle['X_train_raw'][:, 0],
            y=dataset_bundle['X_train_raw'][:, 1],
            mode='markers',
            name='Train split',
            marker=dict(
                color=dataset_bundle['y_train'],
                colorscale='Viridis',
                size=8,
                line=dict(width=1, color='#0f172a'),
                symbol='circle',
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dataset_bundle['X_test_raw'][:, 0],
            y=dataset_bundle['X_test_raw'][:, 1],
            mode='markers',
            name='Test split',
            marker=dict(
                color=dataset_bundle['y_test'],
                colorscale='Viridis',
                size=9,
                line=dict(width=1, color='#0f172a'),
                symbol='diamond',
            ),
        )
    )
    fig.update_layout(
        title=f"Decision boundary for the {dataset_bundle['dataset_name']} classification dataset",
        xaxis_title='x1',
        yaxis_title='x2',
        margin=dict(l=0, r=0, t=48, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


def make_activation_figure(activation):
    z_values = np.linspace(-5, 5, 400)
    if activation == 'relu':
        activations = np.maximum(0, z_values)
        title = 'ReLU activation: max(0, z)'
    elif activation == 'tanh':
        activations = np.tanh(z_values)
        title = 'Tanh activation: tanh(z)'
    elif activation == 'sigmoid':
        activations = 1.0 / (1.0 + np.exp(-z_values))
        title = 'Sigmoid activation: 1 / (1 + e^(-z))'
    else:
        activations = z_values
        title = f'Unknown activation: {activation}'

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=z_values,
            y=activations,
            mode='lines',
            line=dict(color='#0f766e', width=3),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title='z',
        yaxis_title='rho(z)',
        margin=dict(l=0, r=0, t=44, b=0),
    )
    return fig


def make_network_diagram_figure(
    input_dim=2,
    hidden_layers=None,
    output_dim=1,
    params=None,
    activation='tanh',
    feature_names=None,
    activation_snapshot=None,
    previous_weights=None,
    is_training=False,
):
    if hidden_layers is None:
        hidden_layers = [6, 6]

    layer_sizes = [input_dim] + list(hidden_layers) + [output_dim]
    layer_x_positions = np.linspace(0, 1, len(layer_sizes))
    weights, biases = level2_deserialize_params(params) if params else (None, None)
    safe_feature_names = feature_names or [f'x{i + 1}' for i in range(input_dim)]
    layer_activity = activation_snapshot.get('layer_activations') if activation_snapshot else None

    nodes_x = []
    nodes_y = []
    labels = []
    hover_text = []
    colours = []
    node_sizes = []
    edge_traces = []
    layer_coordinates = []
    annotations = []

    for layer_index, layer_size in enumerate(layer_sizes):
        y_positions = np.linspace(0, 1, layer_size)
        layer_coordinates.append(y_positions)

        if layer_index == 0:
            layer_name = 'Feature Layer'
            colour = '#93c5fd'
        elif layer_index == len(layer_sizes) - 1:
            layer_name = 'Output Layer'
            colour = '#fca5a5'
        else:
            layer_name = f'Hidden Layer {layer_index}'
            colour = '#86efac'

        annotations.append(
            dict(
                x=layer_x_positions[layer_index],
                y=1.1,
                text=(
                    f'{layer_name}<br>{layer_size} neuron(s)'
                    if not layer_activity
                    else f'{layer_name}<br>{layer_size} neuron(s)<br>mean |a|={np.mean(np.abs(layer_activity[layer_index])):.2f}'
                ),
                showarrow=False,
                font=dict(size=11),
            )
        )

        for node_index, y_position in enumerate(y_positions):
            nodes_x.append(layer_x_positions[layer_index])
            nodes_y.append(y_position)

            activation_value = None
            if layer_activity and node_index < len(layer_activity[layer_index]):
                activation_value = float(layer_activity[layer_index][node_index])
                colours.append(_activation_to_colour(activation_value))
                node_sizes.append(16 + (12 * min(1.0, abs(activation_value))))
            else:
                colours.append(colour)
                node_sizes.append(16)

            if layer_index == 0:
                feature_label = safe_feature_names[node_index] if node_index < len(safe_feature_names) else f'x{node_index + 1}'
                labels.append(feature_label)
                if activation_value is None:
                    hover_text.append(f'Engineered feature: {feature_label}')
                else:
                    hover_text.append(
                        f'Engineered feature: {feature_label}<br>Probe value={activation_value:+.3f}'
                    )
            elif layer_index == len(layer_sizes) - 1:
                labels.append('yhat' if output_dim == 1 else f'c{node_index}')
                bias_value = biases[layer_index - 1][node_index, 0] if biases is not None else 0.0
                output_name = 'sigmoid' if output_dim == 1 else 'softmax'
                activation_line = '' if activation_value is None else f'<br>Activation={activation_value:+.3f}'
                hover_text.append(
                    f'Output neuron {node_index + 1}<br>bias={bias_value:+.3f}{activation_line}<br>{output_name} head'
                )
            else:
                labels.append(f'h{layer_index}.{node_index + 1}')
                bias_value = biases[layer_index - 1][node_index, 0] if biases is not None else 0.0
                activation_line = '' if activation_value is None else f'<br>Activation={activation_value:+.3f}'
                hover_text.append(f'Layer {layer_index} neuron {node_index + 1}<br>bias={bias_value:+.3f}{activation_line}<br>a = {activation}(z)')

    for layer_index in range(len(layer_sizes) - 1):
        source_x = layer_x_positions[layer_index]
        target_x = layer_x_positions[layer_index + 1]
        source_y_positions = layer_coordinates[layer_index]
        target_y_positions = layer_coordinates[layer_index + 1]
        current_weights = weights[layer_index] if weights is not None else None
        previous_layer_weights = None

        if previous_weights and layer_index < len(previous_weights):
            previous_layer_weights = np.array(previous_weights[layer_index], dtype=np.float64)

        for source_index, source_y in enumerate(source_y_positions):
            for target_index, target_y in enumerate(target_y_positions):
                if current_weights is not None:
                    weight_value = current_weights[target_index, source_index]
                    weight_delta = 0.0
                    if previous_layer_weights is not None:
                        weight_delta = weight_value - previous_layer_weights[target_index, source_index]
                    edge_colour = '#2563eb' if weight_value >= 0 else '#dc2626'
                    edge_width = max(0.6, min(5.5, abs(weight_value) * 2.2 + abs(weight_delta) * 14.0))
                    edge_dash = 'solid'
                    if abs(weight_delta) >= 0.03:
                        edge_dash = 'dash'
                    elif abs(weight_delta) >= 0.01:
                        edge_dash = 'dot'
                    edge_opacity = min(1.0, 0.28 + abs(weight_delta) * 18.0) if previous_layer_weights is not None else 0.62
                    edge_hover = (
                        f'Layer {layer_index + 1} weight[{target_index + 1}, {source_index + 1}] = '
                        f'{weight_value:+.3f}'
                    )
                    if previous_layer_weights is not None:
                        edge_hover += f'<br>Delta this epoch={weight_delta:+.4f}'
                else:
                    edge_colour = '#94a3b8'
                    edge_width = 1
                    edge_dash = 'solid'
                    edge_opacity = 0.4
                    edge_hover = 'Weight'

                edge_traces.append(
                    go.Scatter(
                        x=[source_x, (source_x + target_x) / 2, target_x],
                        y=[source_y, (source_y + target_y) / 2, target_y],
                        mode='lines',
                        line=dict(color=edge_colour, width=edge_width, dash=edge_dash),
                        hovertext=[None, edge_hover, None],
                        hoverinfo='text',
                        showlegend=False,
                        opacity=edge_opacity,
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
        marker=dict(size=node_sizes, color=colours, line=dict(width=1.4, color='#0f172a')),
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    if activation_snapshot:
        probe_raw = activation_snapshot['probe_raw']
        prediction_vector = activation_snapshot['prediction_vector']
        if len(prediction_vector) == 1:
            prediction_text = f"p(class 1)={prediction_vector[0]:.3f}"
        else:
            prediction_text = ', '.join(f'p{index}={value:.3f}' for index, value in enumerate(prediction_vector))

        annotations.append(
            dict(
                x=0.5,
                y=-0.12,
                xref='paper',
                yref='paper',
                text=(
                    f"Probe sample #{activation_snapshot['probe_index'] + 1}: "
                    f"(x1={probe_raw[0]:+.2f}, x2={probe_raw[1]:+.2f}) | "
                    f"target={activation_snapshot['probe_label']} | "
                    f"predicted={activation_snapshot['predicted_label']} | "
                    f"confidence={activation_snapshot['confidence']:.3f} | {prediction_text}"
                ),
                showarrow=False,
                font=dict(size=11, color='#334155'),
            )
        )

    fig.update_layout(
        title='Central FNN architecture view' + (' - animated while training' if is_training else ''),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=0, r=0, t=70, b=36),
        showlegend=False,
        annotations=annotations,
        transition=dict(duration=280, easing='cubic-in-out'),
        uirevision='level2-network-diagram',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
    )
    return fig


def make_level2_training_curves_figure(history):
    epochs = history.get('epochs', [])
    train_loss = history.get('train_loss', [])
    test_loss = history.get('test_loss', [])
    train_accuracy = history.get('train_accuracy', [])
    test_accuracy = history.get('test_accuracy', [])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_loss,
            mode='lines+markers',
            name='Train Loss',
            line=dict(color='#dc2626', width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=test_loss,
            mode='lines+markers',
            name='Test Loss',
            line=dict(color='#f97316', width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_accuracy,
            mode='lines+markers',
            name='Train Accuracy',
            line=dict(color='#0f766e', width=3),
            yaxis='y2',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=test_accuracy,
            mode='lines+markers',
            name='Test Accuracy',
            line=dict(color='#2563eb', width=3),
            yaxis='y2',
        )
    )
    fig.update_layout(
        title='Optimisation behaviour across training',
        xaxis_title='Epoch',
        yaxis=dict(title='Loss'),
        yaxis2=dict(title='Accuracy', overlaying='y', side='right', range=[0, 1]),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


def make_level2_output_panel(metrics, dataset_name):
    card_style = {
        'backgroundColor': '#f8fafc',
        'borderRadius': '14px',
        'padding': '14px 16px',
        'border': '1px solid #dbeafe',
    }
    label_style = {
        'fontSize': '12px',
        'textTransform': 'uppercase',
        'letterSpacing': '0.08em',
        'color': '#64748b',
        'marginBottom': '6px',
    }
    value_style = {'fontSize': '24px', 'fontWeight': '700', 'color': '#0f172a'}

    return html.Div([
        html.Div(
            f'{dataset_name.title()} is being treated as a binary classification dataset.',
            style={
                'marginBottom': '14px',
                'color': '#334155',
                'fontWeight': '600',
            }
        ),
        html.Div([
            html.Div([
                html.Div('Train Accuracy', style=label_style),
                html.Div(f"{metrics['train_accuracy'] * 100:.1f}%", style=value_style),
            ], style=card_style),
            html.Div([
                html.Div('Train Loss', style=label_style),
                html.Div(f"{metrics['train_loss']:.4f}", style=value_style),
            ], style=card_style),
            html.Div([
                html.Div('Test Accuracy', style=label_style),
                html.Div(f"{metrics['test_accuracy'] * 100:.1f}%", style=value_style),
            ], style=card_style),
            html.Div([
                html.Div('Test Loss', style=label_style),
                html.Div(f"{metrics['test_loss']:.4f}", style=value_style),
            ], style=card_style),
        ], style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fit, minmax(150px, 1fr))',
            'gap': '12px',
        }),
        html.Div([
            html.Div('Epochs trained', style=label_style),
            html.Div(str(metrics['epoch']), style={'fontWeight': '700', 'fontSize': '18px'}),
            html.Div('Parameters', style={**label_style, 'marginTop': '10px'}),
            html.Div(str(metrics['parameter_count']), style={'fontWeight': '700', 'fontSize': '18px'}),
        ], style={'marginTop': '14px', 'color': '#334155'}),
    ])


def make_level2_summary_panel(params, activation):
    meta = params.get('meta', {})
    input_dim = meta.get('input_dim', 2)
    hidden_layers = meta.get('hidden_layer_sizes', [6, 6])
    output_dim = meta.get('output_dim', 1)
    feature_names = meta.get('feature_names', FEATURE_NAMES[:input_dim])
    layer_sizes = [input_dim] + list(hidden_layers) + [output_dim]
    weights, biases = level2_deserialize_params(params)

    layer_items = []
    for layer_index, (weight, bias) in enumerate(zip(weights, biases), start=1):
        if layer_index < len(weights):
            layer_label = f'Hidden {layer_index}'
            transform = activation
        else:
            layer_label = 'Output'
            transform = 'sigmoid' if output_dim == 1 else 'softmax'

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
            'Forward map: h(l) = rho(W(l)h(l-1) + b(l)), ending in a binary classification head.',
            style={'fontSize': '13px', 'marginBottom': '8px'},
        ),
        html.P(
            f"Feature layer uses: {', '.join(feature_names)}",
            style={'fontSize': '13px', 'color': '#475569', 'marginBottom': '8px'},
        ),
        html.P(
            f'Hidden activation: {activation} | Total parameters: {level2_parameter_count(params)}',
            style={'fontSize': '13px', 'color': '#475569', 'marginBottom': '10px'},
        ),
        html.Ul(layer_items, style={'paddingLeft': '18px', 'marginBottom': '0'}),
    ])