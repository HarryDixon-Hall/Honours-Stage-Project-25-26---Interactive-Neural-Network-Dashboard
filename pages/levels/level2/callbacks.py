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
    apply_level2_gradients,
    build_level2_dataset,
    compute_level2_gradients,
    init_level2_mlp,
    level2_evaluate_metrics,
    make_level2_activation_snapshot,
    make_level2_boundary_explanation,
    make_level2_training_stage_panel,
    level2_set_baseline_history,
    make_activation_figure,
    make_decision_boundary_figure,
    make_level2_output_panel,
    make_level2_summary_panel,
    make_network_diagram_figure,
)


MAX_LEVEL2_EPOCHS = 200
SPEED_TO_INTERVAL = {
    'slow': 700,
    'normal': 350,
    'fast': 180,
}
STAGE_SEQUENCE = ('forward', 'loss', 'backward', 'update')
VISIBLE_WRAPPER_STYLE = {'flex': '1 1 200px'}
HIDDEN_WRAPPER_STYLE = {'flex': '1 1 200px', 'display': 'none'}


def _copy_serialized_weights(weights):
    return [[list(row) for row in layer] for layer in weights]


def _copy_params(params):
    if params is None:
        return None

    copied = {
        'weights': _copy_serialized_weights(params.get('weights', [])),
        'biases': _copy_serialized_weights(params.get('biases', [])),
        'epoch': int(params.get('epoch', 0)),
        'history': {
            key: list(value)
            for key, value in (params.get('history') or {}).items()
        },
    }

    if params.get('meta') is not None:
        copied['meta'] = dict(params['meta'])

    return copied


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


def _make_training_state(play_speed='normal', mode='auto'):
    safe_mode = mode if mode in {'auto', 'semiauto'} else 'auto'
    return {
        'running': False,
        'paused': False,
        'max_epochs': MAX_LEVEL2_EPOCHS,
        'probe_index': 0,
        'previous_weights': None,
        'play_speed': play_speed,
        'mode': safe_mode,
        'stage_index': 0,
        'current_stage': 'idle',
        'pending_gradients': None,
        'pending_loss': None,
        'gradient_norms': None,
        'epoch_base_params': None,
        'epoch_updated_params': None,
    }


def _ensure_epoch_cycle(training_state, dataset_bundle, params, activation, learning_rate, meta):
    if training_state.get('epoch_base_params') and training_state.get('epoch_updated_params'):
        return training_state

    base_params = _copy_params(params)
    base_params['meta'] = dict(meta)
    probe_index = int(base_params.get('epoch', 0)) % max(1, dataset_bundle['X_train'].shape[0])
    gradient_snapshot = compute_level2_gradients(
        dataset_bundle,
        base_params,
        activation=activation,
        l2=1e-4,
    )
    updated_params = apply_level2_gradients(
        dataset_bundle,
        _copy_params(base_params),
        gradient_snapshot,
        activation=activation,
        lr=learning_rate,
        l2=1e-4,
    )
    updated_params['meta'] = dict(meta)

    training_state['epoch_base_params'] = base_params
    training_state['epoch_updated_params'] = updated_params
    training_state['probe_index'] = probe_index
    training_state['pending_gradients'] = gradient_snapshot
    training_state['pending_loss'] = gradient_snapshot.get('loss')
    training_state['gradient_norms'] = gradient_snapshot.get('gradient_norms')

    return training_state


def _apply_stage_view(training_state, meta):
    stage_index = int(training_state.get('stage_index', 0)) % len(STAGE_SEQUENCE)
    current_stage = STAGE_SEQUENCE[stage_index]
    training_state['current_stage'] = current_stage

    base_params = training_state.get('epoch_base_params')
    updated_params = training_state.get('epoch_updated_params')

    if current_stage == 'update' and updated_params is not None:
        display_params = _copy_params(updated_params)
        training_state['previous_weights'] = _copy_serialized_weights(base_params.get('weights', [])) if base_params else None
    else:
        display_params = _copy_params(base_params)
        training_state['previous_weights'] = None

    if display_params is not None:
        display_params['meta'] = dict(meta)

    return display_params, training_state


def _clear_epoch_cycle(training_state):
    training_state['epoch_base_params'] = None
    training_state['epoch_updated_params'] = None
    training_state['previous_weights'] = None
    training_state['pending_gradients'] = None
    training_state['pending_loss'] = None
    training_state['gradient_norms'] = None
    return training_state


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
        Output('level2-train-toggle-btn', 'disabled'),
        Output('level2-pause-btn', 'disabled'),
        Output('level2-prev-stage-btn', 'disabled'),
        Output('level2-step-btn', 'disabled'),
        Output('level2-train-interval', 'disabled'),
        Output('level2-train-interval', 'interval'),
        Input('level2-train-toggle-btn', 'n_clicks'),
        Input('level2-pause-btn', 'n_clicks'),
        Input('level2-prev-stage-btn', 'n_clicks'),
        Input('level2-step-btn', 'n_clicks'),
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
        Input('level2-play-speed-control', 'value'),
        Input('level2-training-mode', 'value'),
        State('level2-params-store', 'data'),
        State('level2-training-store', 'data'),
    )
    def update_level2_state(
        n_train_toggle,
        n_pause,
        n_prev_stage,
        n_step,
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
        play_speed,
        training_mode,
        params,
        training_store,
    ):
        _ = (n_train_toggle, n_pause, n_prev_stage, n_step, n_reset, n_intervals)
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
        safe_play_speed = play_speed if play_speed in SPEED_TO_INTERVAL else 'normal'
        safe_mode = training_mode if training_mode in {'auto', 'semiauto'} else 'auto'
        training_state = training_store or _make_training_state(play_speed='normal', mode=safe_mode)
        training_state['play_speed'] = safe_play_speed
        training_state['mode'] = safe_mode
        running = bool(training_state.get('running'))
        paused = bool(training_state.get('paused'))

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
            paused = False
            training_state = _make_training_state(play_speed=safe_play_speed, mode=safe_mode)
        elif trigger == 'level2-train-toggle-btn':
            if params.get('epoch', 0) < MAX_LEVEL2_EPOCHS:
                running = True
                paused = False
                training_state = _ensure_epoch_cycle(
                    training_state,
                    dataset_bundle,
                    params,
                    activation,
                    safe_learning_rate,
                    meta,
                )
                if training_state.get('current_stage') == 'idle':
                    training_state['stage_index'] = 0
                params, training_state = _apply_stage_view(training_state, meta)
        elif trigger == 'level2-pause-btn':
            if running:
                running = False
                paused = True
        elif trigger in {'level2-train-interval', 'level2-step-btn', 'level2-prev-stage-btn'} and running:
            allow_auto_advance = trigger == 'level2-train-interval' and safe_mode == 'auto'
            allow_next = trigger == 'level2-step-btn' and safe_mode == 'semiauto'
            allow_prev = trigger == 'level2-prev-stage-btn' and safe_mode == 'semiauto'

            if allow_auto_advance or allow_next or allow_prev:
                training_state = _ensure_epoch_cycle(
                    training_state,
                    dataset_bundle,
                    params,
                    activation,
                    safe_learning_rate,
                    meta,
                )
                stage_index = int(training_state.get('stage_index', 0))

                if allow_prev:
                    training_state['stage_index'] = max(0, stage_index - 1)
                    params, training_state = _apply_stage_view(training_state, meta)
                else:
                    if stage_index < len(STAGE_SEQUENCE) - 1:
                        training_state['stage_index'] = stage_index + 1
                        params, training_state = _apply_stage_view(training_state, meta)
                    else:
                        params = _copy_params(training_state.get('epoch_updated_params')) or params
                        params['meta'] = meta

                        if params.get('epoch', 0) >= MAX_LEVEL2_EPOCHS:
                            running = False
                            paused = False
                            training_state = _clear_epoch_cycle(training_state)
                            training_state['stage_index'] = 0
                            training_state['current_stage'] = 'idle'
                        else:
                            training_state = _clear_epoch_cycle(training_state)
                            training_state = _ensure_epoch_cycle(
                                training_state,
                                dataset_bundle,
                                params,
                                activation,
                                safe_learning_rate,
                                meta,
                            )
                            training_state['stage_index'] = 0
                            params, training_state = _apply_stage_view(training_state, meta)

        if params.get('epoch', 0) >= MAX_LEVEL2_EPOCHS:
            running = False
            paused = False
            training_state = _clear_epoch_cycle(training_state)
            training_state['current_stage'] = 'idle'
            training_state['stage_index'] = 0

        if not running and not paused and params.get('epoch', 0) == 0 and int(training_state.get('stage_index', 0)) == 0:
            training_state['current_stage'] = 'idle'

        params['meta'] = meta
        training_state = {
            'running': running,
            'paused': paused,
            'max_epochs': MAX_LEVEL2_EPOCHS,
            'probe_index': training_state.get('probe_index', 0),
            'previous_weights': training_state.get('previous_weights'),
            'play_speed': safe_play_speed,
            'mode': safe_mode,
            'stage_index': training_state.get('stage_index', 0),
            'current_stage': training_state.get('current_stage', 'idle' if not running else 'forward'),
            'pending_gradients': training_state.get('pending_gradients'),
            'pending_loss': training_state.get('pending_loss'),
            'gradient_norms': training_state.get('gradient_norms'),
            'epoch_base_params': training_state.get('epoch_base_params'),
            'epoch_updated_params': training_state.get('epoch_updated_params'),
        }
        max_epochs_reached = params.get('epoch', 0) >= MAX_LEVEL2_EPOCHS
        if max_epochs_reached:
            button_label = 'Max Epochs Reached'
        elif running:
            button_label = 'Training Active'
        elif paused or params.get('epoch', 0) > 0 or training_state.get('current_stage') != 'idle':
            button_label = 'Resume Training'
        else:
            button_label = 'Start Training'
        start_disabled = running or max_epochs_reached
        pause_disabled = (not running) or max_epochs_reached
        prev_stage_disabled = (not running) or safe_mode != 'semiauto' or int(training_state.get('stage_index', 0)) <= 0 or max_epochs_reached
        step_disabled = (not running) or safe_mode != 'semiauto' or max_epochs_reached
        interval_disabled = (not running) or safe_mode != 'auto' or max_epochs_reached
        interval_ms = SPEED_TO_INTERVAL[safe_play_speed]

        return params, training_state, button_label, start_disabled, pause_disabled, prev_stage_disabled, step_disabled, interval_disabled, interval_ms

    @app.callback(
        Output('level2-network-diagram-graph', 'figure'),
        Output('level2-math-explanation', 'children'),
        Output('level2-output-summary', 'children'),
        Output('level2-decision-boundary-graph', 'figure'),
        Output('level2-boundary-explanation', 'children'),
        Output('level2-activation-graph', 'figure'),
        Output('level2-epoch-live', 'children'),
        Output('level2-training-stage-panel', 'children'),
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
        training_state = training_store or {}
        probe_index = training_state.get('probe_index', 0)
        previous_weights = training_state.get('previous_weights')
        current_stage = training_state.get('current_stage', 'idle')
        pending_gradients = training_state.get('pending_gradients')
        pending_loss = training_state.get('pending_loss')
        gradient_norms = training_state.get('gradient_norms') or []
        paused = bool(training_state.get('paused'))
        mode = training_state.get('mode', 'auto')
        stage_index = int(training_state.get('stage_index', 0))
        previous_stage = STAGE_SEQUENCE[stage_index - 1] if 0 < stage_index < len(STAGE_SEQUENCE) else 'none'
        next_stage = STAGE_SEQUENCE[(stage_index + 1) % len(STAGE_SEQUENCE)] if current_stage != 'idle' and metrics['epoch'] < MAX_LEVEL2_EPOCHS else STAGE_SEQUENCE[0]
        epoch_base_params = training_state.get('epoch_base_params') or params
        epoch_in_progress = int(epoch_base_params.get('epoch', metrics['epoch'])) + (0 if current_stage == 'idle' else 1)
        activation_snapshot = make_level2_activation_snapshot(
            dataset_bundle,
            params,
            activation,
            probe_index=probe_index,
        )

        fig_network = make_network_diagram_figure(
            input_dim=input_dim,
            hidden_layers=hidden_layer_sizes,
            output_dim=output_dim,
            params=params,
            activation=activation,
            feature_names=feature_names,
            activation_snapshot=activation_snapshot,
            previous_weights=previous_weights,
            is_training=bool(training_state.get('running')),
            stage_name=current_stage,
            gradient_snapshot=pending_gradients,
        )
        explanation = make_level2_summary_panel(params, activation)
        output_summary = make_level2_output_panel(metrics, dataset)
        fig_boundary = make_decision_boundary_figure(dataset_bundle, params, activation)
        boundary_explanation = make_level2_boundary_explanation(dataset, activation_snapshot)
        fig_activation = make_activation_figure(activation)

        running = bool(training_state.get('running'))
        play_speed = training_state.get('play_speed', 'normal')
        if running:
            status_label = 'Running'
            status_color = '#0f766e'
        elif paused:
            status_label = 'Paused'
            status_color = '#b45309'
        else:
            status_label = 'Stopped'
            status_color = '#475569'
        epoch_panel = [
            html.Div([
                html.Div('Live Epoch Count', style={'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '0.08em', 'color': '#64748b'}),
                html.Div(str(metrics['epoch']), style={'fontSize': '28px', 'fontWeight': '700', 'color': '#0f172a'}),
            ]),
            html.Div([
                html.Div('Status', style={'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '0.08em', 'color': '#64748b'}),
                html.Div(status_label, style={'fontSize': '16px', 'fontWeight': '700', 'color': status_color}),
                html.Div(f'Learning rate {learning_rate:.2f}', style={'fontSize': '12px', 'color': '#64748b', 'marginTop': '4px'}),
                html.Div(f'Epoch timeline {epoch_in_progress}', style={'fontSize': '12px', 'color': '#64748b', 'marginTop': '4px'}),
                html.Div(f'Training mode {mode}', style={'fontSize': '12px', 'color': '#64748b', 'marginTop': '4px'}),
                html.Div(f'Play speed {play_speed}', style={'fontSize': '12px', 'color': '#64748b', 'marginTop': '4px'}),
                html.Div(f'Current stage {current_stage}', style={'fontSize': '12px', 'color': '#64748b', 'marginTop': '4px'}),
                html.Div(f'Previous stage {previous_stage}', style={'fontSize': '12px', 'color': '#64748b', 'marginTop': '4px'}),
                html.Div(f'Next stage {next_stage}', style={'fontSize': '12px', 'color': '#64748b', 'marginTop': '4px'}),
                html.Div(
                    (
                        'Animation shows a probe sample flowing through the current network.'
                        if activation_snapshot
                        else 'Animation snapshot unavailable.'
                    ),
                    style={'fontSize': '12px', 'color': '#64748b', 'marginTop': '4px'},
                ),
                html.Div(
                    'Semi-auto mode walks a cached epoch timeline with Previous Stage and Next Stage. Auto mode locks those buttons and advances the same timeline continuously.' if mode == 'semiauto' else 'Auto mode locks manual stepping and advances the cached forward, loss, backward, and update stages continuously before starting the next epoch.',
                    style={'fontSize': '12px', 'color': '#64748b', 'marginTop': '4px'},
                ),
            ]),
        ]

        stage_details = {
            'forward': (
                f"Probe sample {probe_index + 1} moving through the network."
                if activation_snapshot else 'Preparing probe sample activations.'
            ),
            'loss': (
                f"Train loss for the current epoch step: {pending_loss:.4f}."
                if pending_loss is not None else 'Comparing prediction with the target label.'
            ),
            'backward': (
                'Gradient norms: ' + ', '.join(f'L{index + 1}={value:.3f}' for index, value in enumerate(gradient_norms))
                if gradient_norms else 'Backpropagating error signals through the network.'
            ),
            'update': f'Applying gradient descent with learning rate {learning_rate:.2f}.',
        }
        stage_panel = make_level2_training_stage_panel(
            current_stage=current_stage,
            epoch=metrics['epoch'],
            stage_details=stage_details,
        )

        return (
            fig_network,
            explanation,
            output_summary,
            fig_boundary,
            boundary_explanation,
            fig_activation,
            epoch_panel,
            stage_panel,
        )