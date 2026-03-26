from importlib import import_module

import dash

try:
    from dash import Input, Output, State
except ImportError:
    dash_dependencies = import_module('dash.dependencies')
    Input = dash_dependencies.Input
    Output = dash_dependencies.Output
    State = dash_dependencies.State

from pages.levels.level2.methods import (
    init_level2_mlp,
    level2_evaluate_metrics,
    level2_set_baseline_history,
    load_toy_dataset,
    make_activation_figure,
    make_decision_boundary_figure,
    make_level2_comparison_panel,
    make_level2_metrics_cards,
    make_level2_summary_panel,
    make_level2_training_curves_figure,
    make_network_diagram_figure,
    train_level2_model,
)


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
    def update_level2_params(
        n_train,
        n_reset,
        n_compare,
        hidden_layers,
        neurons_per_layer,
        activation,
        dataset,
        params,
        compare_store,
    ):
        _ = (n_train, n_reset, n_compare)
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
        Input('level2-compare-store', 'data'),
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
                f"{dataset.title()} dataset | Architecture {' -> '.join(str(size) for size in meta.get('layer_sizes', [2] + hidden_layer_sizes + [1]))}"
            )
        )

        fig_activation = make_activation_figure(activation)
        fig_network = make_network_diagram_figure(
            input_dim=2,
            hidden_layers=hidden_layer_sizes,
            output_dim=1,
            params=params,
            activation=activation,
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
