from importlib import import_module

import dash

try:
    from dash import Input, Output, State
except ImportError:
    dash_dependencies = import_module('dash.dependencies')
    Input = dash_dependencies.Input
    Output = dash_dependencies.Output
    State = dash_dependencies.State

from dash import html

from pages.levels.level2.methods import (
    build_level2_dataset,
    init_level2_mlp,
    level2_evaluate_metrics,
    level2_set_baseline_history,
    make_activation_figure,
    make_decision_boundary_figure,
    make_level2_output_panel,
    make_level2_summary_panel,
    make_network_diagram_figure,
    train_level2_model,
)


MAX_LEVEL2_EPOCHS = 200
VISIBLE_WRAPPER_STYLE = {'flex': '1 1 200px'}
HIDDEN_WRAPPER_STYLE = {'flex': '1 1 200px', 'display': 'none'}


def _safe_int(value, minimum, maximum, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _safe_float(value, minimum, maximum, default):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _build_meta(dataset_bundle, input_dim, hidden_layer_sizes, output_dim, activation, learning_rate, dataset):
    return {
        'dataset': dataset,
        'activation': activation,
        'input_dim': input_dim,
        'hidden_layer_sizes': hidden_layer_sizes,
        'hidden_layers': len(hidden_layer_sizes),
        'output_dim': output_dim,
        'learning_rate': learning_rate,
        'layer_sizes': [input_dim] + hidden_layer_sizes + [output_dim],
        'feature_names': dataset_bundle['feature_names'],
    }


def register_level2_callbacks(app):
    @app.callback(
        Output('level2-hidden-layer-1-wrapper', 'style'),
        Output('level2-hidden-layer-2-wrapper', 'style'),
        Output('level2-hidden-layer-3-wrapper', 'style'),
        Output('level2-hidden-layer-4-wrapper', 'style'),
        Input('level2-hidden-layers-slider', 'value'),
    )
    def update_hidden_layer_visibility(hidden_layers):
        safe_hidden_layers = _safe_int(hidden_layers, 1, 4, 2)
        styles = []

        for layer_index in range(1, 5):
            styles.append(VISIBLE_WRAPPER_STYLE if layer_index <= safe_hidden_layers else HIDDEN_WRAPPER_STYLE)

        return tuple(styles)

    @app.callback(
        Output('level2-params-store', 'data'),
        Output('level2-training-store', 'data'),
        Output('level2-train-toggle-btn', 'children'),
        Output('level2-train-interval', 'disabled'),
        Input('level2-train-toggle-btn', 'n_clicks'),
        Input('level2-reset-btn', 'n_clicks'),
        Input('level2-train-interval', 'n_intervals'),
        Input('level2-input-dim-input', 'value'),
        Input('level2-hidden-layers-slider', 'value'),
        Input('level2-hidden-layer-1-input', 'value'),
        Input('level2-hidden-layer-2-input', 'value'),
        Input('level2-hidden-layer-3-input', 'value'),
        Input('level2-hidden-layer-4-input', 'value'),
        Input('level2-output-dim-input', 'value'),
        Input('level2-activation-dropdown', 'value'),
        Input('level2-dataset-dropdown', 'value'),
        Input('level2-learning-rate-slider', 'value'),
        State('level2-params-store', 'data'),
        State('level2-training-store', 'data'),
    )
    def update_level2_state(
        n_train_toggle,
        n_reset,
        n_intervals,
        input_dim,
        hidden_layers,
        hidden_1,
        hidden_2,
        hidden_3,
        hidden_4,
        output_dim,
        activation,
        dataset,
        learning_rate,
        params,
        training_store,
    ):
        _ = (n_train_toggle, n_reset, n_intervals)
        ctx = dash.callback_context
        trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        safe_input_dim = _safe_int(input_dim, 2, 8, 2)
        safe_hidden_layers = _safe_int(hidden_layers, 1, 4, 2)
        hidden_candidates = [hidden_1, hidden_2, hidden_3, hidden_4]
        hidden_layer_sizes = [
            _safe_int(hidden_candidates[layer_index], 2, 16, 6 if layer_index < 2 else 4)
            for layer_index in range(safe_hidden_layers)
        ]
        safe_output_dim = _safe_int(output_dim, 1, 2, 1)
        safe_learning_rate = _safe_float(learning_rate, 0.01, 0.2, 0.08)
        training_state = training_store or {'running': False}
        running = bool(training_state.get('running'))

        dataset_bundle = build_level2_dataset(dataset, input_dim=safe_input_dim)
        meta = _build_meta(
            dataset_bundle,
            safe_input_dim,
            hidden_layer_sizes,
            safe_output_dim,
            activation,
            safe_learning_rate,
            dataset,
        )

        rebuild_triggers = {
            None,
            'level2-reset-btn',
            'level2-input-dim-input',
            'level2-hidden-layers-slider',
            'level2-hidden-layer-1-input',
            'level2-hidden-layer-2-input',
            'level2-hidden-layer-3-input',
            'level2-hidden-layer-4-input',
            'level2-output-dim-input',
            'level2-activation-dropdown',
            'level2-dataset-dropdown',
        }

        if params is None or trigger in rebuild_triggers:
            params = init_level2_mlp(
                input_dim=safe_input_dim,
                hidden_layers=hidden_layer_sizes,
                output_dim=safe_output_dim,
            )
            params['meta'] = meta
            params = level2_set_baseline_history(dataset_bundle, params, activation, l2=1e-4)
            running = False
        elif trigger == 'level2-train-toggle-btn':
            running = not running
        elif trigger == 'level2-train-interval' and running:
            params['meta'] = meta
            params = train_level2_model(
                dataset_bundle,
                params,
                activation=activation,
                epochs=1,
                lr=safe_learning_rate,
                l2=1e-4,
            )
            if params.get('epoch', 0) >= MAX_LEVEL2_EPOCHS:
                running = False

        params['meta'] = meta
        training_state = {
            'running': running,
            'max_epochs': MAX_LEVEL2_EPOCHS,
        }
        button_label = 'Stop Training' if running else 'Start Training'
        interval_disabled = not running

        return params, training_state, button_label, interval_disabled

    @app.callback(
        Output('level2-network-diagram-graph', 'figure'),
        Output('level2-math-explanation', 'children'),
        Output('level2-output-summary', 'children'),
        Output('level2-decision-boundary-graph', 'figure'),
        Output('level2-activation-graph', 'figure'),
        Output('level2-epoch-live', 'children'),
        Input('level2-params-store', 'data'),
        Input('level2-training-store', 'data'),
    )
    def update_level2_views(params, training_store):
        if params is None:
            raise dash.exceptions.PreventUpdate

        meta = params.get('meta', {})
        activation = meta.get('activation', 'tanh')
        dataset = meta.get('dataset', 'moons')
        input_dim = meta.get('input_dim', 2)
        hidden_layer_sizes = meta.get('hidden_layer_sizes', [6, 6])
        output_dim = meta.get('output_dim', 1)
        feature_names = meta.get('feature_names', [])
        learning_rate = meta.get('learning_rate', 0.08)
        dataset_bundle = build_level2_dataset(dataset, input_dim=input_dim)
        metrics = level2_evaluate_metrics(dataset_bundle, params, activation, l2=1e-4)

        fig_network = make_network_diagram_figure(
            input_dim=input_dim,
            hidden_layers=hidden_layer_sizes,
            output_dim=output_dim,
            params=params,
            activation=activation,
            feature_names=feature_names,
        )
        explanation = make_level2_summary_panel(params, activation)
        output_summary = make_level2_output_panel(metrics, dataset)
        fig_boundary = make_decision_boundary_figure(dataset_bundle, params, activation)
        fig_activation = make_activation_figure(activation)

        running = bool((training_store or {}).get('running'))
        epoch_panel = [
            html.Div([
                html.Div('Live Epoch Count', style={'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '0.08em', 'color': '#64748b'}),
                html.Div(str(metrics['epoch']), style={'fontSize': '28px', 'fontWeight': '700', 'color': '#0f172a'}),
            ]),
            html.Div([
                html.Div('Status', style={'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '0.08em', 'color': '#64748b'}),
                html.Div('Running' if running else 'Stopped', style={'fontSize': '16px', 'fontWeight': '700', 'color': '#0f766e' if running else '#475569'}),
                html.Div(f'Learning rate {learning_rate:.2f}', style={'fontSize': '12px', 'color': '#64748b', 'marginTop': '4px'}),
            ]),
        ]

        return (
            fig_network,
            explanation,
            output_summary,
            fig_boundary,
            fig_activation,
            epoch_panel,
        )