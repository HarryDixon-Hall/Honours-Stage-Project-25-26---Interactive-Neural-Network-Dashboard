from importlib import import_module

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


def build_architecture_diagram(model, hidden_size):
    fig = go.Figure()

    input_nodes = [index / 5 for index in range(4)]
    hidden_nodes = [index / 16 for index in range(hidden_size)]
    output_nodes = [index / 3 for index in range(3)]

    fig.add_trace(go.Scatter(
        x=[0] * 4,
        y=input_nodes,
        mode='markers+text',
        marker=dict(size=20, color='blue'),
        text=[f'I{index + 1}' for index in range(4)],
        name='Input'
    ))
    fig.add_trace(go.Scatter(
        x=[1] * hidden_size,
        y=hidden_nodes,
        mode='markers+text',
        marker=dict(size=15, color='orange'),
        text=[f'H{index + 1}' for index in range(hidden_size)],
        name='Hidden'
    ))
    fig.add_trace(go.Scatter(
        x=[2] * 3,
        y=output_nodes,
        mode='markers+text',
        marker=dict(size=20, color='green'),
        text=['O1', 'O2', 'O3'],
        name='Output'
    ))

    for input_index in range(4):
        for hidden_index in range(hidden_size):
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[input_nodes[input_index], hidden_nodes[hidden_index]],
                mode='lines',
                line=dict(width=1, color='gray'),
                showlegend=False,
                hoverinfo='skip'
            ))

    fig.update_layout(
        title=f'Live Architecture: 4 → {hidden_size} → 3',
        xaxis=dict(showgrid=False, range=[-0.2, 2.2]),
        yaxis=dict(showgrid=False, range=[-0.2, 1.2]),
        height=400,
        showlegend=False,
    )
    return fig


def plot_decision_boundary(model, Xtrain_sample):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(Xtrain_sample[:100])

    step = 0.02
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    grid_input = np.c_[xx.ravel(), yy.ravel(), np.zeros((xx.ravel().shape[0], 2))]
    predictions = model.forward(grid_input)
    decision = np.argmax(predictions, axis=1).reshape(xx.shape)

    fig = go.Figure(data=go.Heatmap(
        z=decision,
        x=xx[0],
        y=yy[:, 0],
        colorscale='RdYlBu',
        hoverongaps=False,
    ))
    fig.update_layout(
        title='Live Decision Boundary (PCA Projection)',
        xaxis_title='PC1',
        yaxis_title='PC2',
        height=350,
    )
    return fig


def generate_data(dataset):
    if dataset == 'linear':
        X, y = make_blobs(
            n_samples=300,
            centers=[(-2, -2), (2, 2)],
            cluster_std=0.8,
            random_state=42,
        )
    elif dataset == 'moons':
        X, y = make_moons(
            n_samples=300,
            noise=0.2,
            random_state=42,
        )
    elif dataset == 'circles':
        X, y = make_circles(
            n_samples=300,
            noise=0.1,
            factor=0.4,
            random_state=42,
        )
    else:
        X, y = make_blobs(
            n_samples=300,
            centers=[(-2, -2), (2, 2)],
            cluster_std=0.8,
            random_state=42,
        )

    return X, y


def register_level1_callbacks(app):
    @app.callback(
        [Output('live-architecture', 'figure'), Output('live-weights-heatmap', 'figure'),
         Output('code-preview', 'children')],
        [Input('live-hidden-size', 'value'), Input('live-seed', 'value')]
    )
    def update_live_visualisations(hidden_size, seed):
        Xtrain, _, _, _, _ = load_dataset('iris')

        np.random.seed(seed or 42)
        model = SimpleNN(4, hidden_size, 3)

        arch_fig = build_architecture_diagram(model, hidden_size)

        weight_fig = go.Figure()
        weight_fig.add_trace(go.Heatmap(z=model.W1, colorscale='RdBu', zmid=0))
        weight_fig.update_layout(title=f'W1 Weights (4×{hidden_size})')

        code_preview = html.Textarea(
            value=f'model = SimpleNN(4, {hidden_size}, 3, seed={seed})',
            style={'width': '100%', 'height': 100}
        )

        _ = Xtrain
        return arch_fig, weight_fig, code_preview

    @app.callback(
        Output('l1-decision-boundary', 'figure'),
        [Input('l1-dataset', 'value')]
    )
    def update_decision_boundary(dataset):
        X, y = generate_data(dataset)
        x1 = X[:, 0]
        x2 = X[:, 1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x1[y == 0],
            y=x2[y == 0],
            mode='markers',
            name='Class 0',
            marker=dict(color='blue', size=8, opacity=0.7),
        ))
        fig.add_trace(go.Scatter(
            x=x1[y == 1],
            y=x2[y == 1],
            mode='markers',
            name='Class 1',
            marker=dict(color='red', size=8, opacity=0.7),
        ))

        x_min, x_max = x1.min() - 0.5, x1.max() + 0.5
        y_min, y_max = x2.min() - 0.5, x2.max() + 0.5
        fig.update_layout(
            title=f'{dataset.title()} dataset preview',
            xaxis_title='x₁',
            yaxis_title='x₂',
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max], scaleanchor='x', scaleratio=1),
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=40, r=10, t=40, b=40),
        )
        return fig