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

from modelFactory.dataload import load_dataset
from modelFactory.models import SimpleNN
from pages.levels.level1.methods import build_architecture_diagram, generate_data


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