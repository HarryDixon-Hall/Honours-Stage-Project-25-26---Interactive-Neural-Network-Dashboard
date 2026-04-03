from importlib import import_module
from urllib.parse import parse_qs

import dash

try:
    from dash import Input, Output, State
except ImportError:
    dash_dependencies = import_module('dash.dependencies')
    Input = dash_dependencies.Input
    Output = dash_dependencies.Output
    State = dash_dependencies.State

from dash import html

from distribution.database import get_level2_model_run, save_level2_model_run

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
VISIBLE_WRAPPER_STYLE = {'flex': '1 1 110px', 'minWidth': '110px'}
HIDDEN_WRAPPER_STYLE = {'flex': '1 1 110px', 'minWidth': '110px', 'display': 'none'}


def _copy_serialized_weights(weights):
    return [[list(row) for row in layer] for layer in weights]


def _copy_replay_frames(replay_frames):
    copied = []
    for frame in replay_frames or []:
        copied.append({
            'epoch': int(frame.get('epoch', 0)),
            'weights': _copy_serialized_weights(frame.get('weights', [])),
            'biases': _copy_serialized_weights(frame.get('biases', [])),
            'history': {
                key: list(value)
                for key, value in (frame.get('history') or {}).items()
            },
            'meta': dict(frame.get('meta', {})),
        })
    return copied


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

    if params.get('replay_frames') is not None:
        copied['replay_frames'] = _copy_replay_frames(params.get('replay_frames', []))

    if params.get('saved_run_id') is not None:
        copied['saved_run_id'] = int(params['saved_run_id'])

    if params.get('saved_model_name') is not None:
        copied['saved_model_name'] = params['saved_model_name']

    return copied


def _build_params_from_replay_frame(frame, replay_frames, *, saved_run_id=None, saved_model_name=None):
    params = {
        'weights': _copy_serialized_weights(frame.get('weights', [])),
        'biases': _copy_serialized_weights(frame.get('biases', [])),
        'epoch': int(frame.get('epoch', 0)),
        'history': {
            key: list(value)
            for key, value in (frame.get('history') or {}).items()
        },
        'meta': dict(frame.get('meta', {})),
        'replay_frames': _copy_replay_frames(replay_frames),
    }

    if saved_run_id is not None:
        params['saved_run_id'] = int(saved_run_id)

    if saved_model_name is not None:
        params['saved_model_name'] = saved_model_name

    return params


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
        Input('level2-replay-store', 'data'),
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
        replay_store,
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
        replay_active = bool((replay_store or {}).get('active'))
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
        if replay_active:
            button_label = 'Replay Active'
        elif max_epochs_reached:
            button_label = 'Max Epochs Reached'
        elif running:
            button_label = 'Training Active'
        elif paused or params.get('epoch', 0) > 0 or training_state.get('current_stage') != 'idle':
            button_label = 'Resume Training'
        else:
            button_label = 'Start Training'
        start_disabled = replay_active or running or max_epochs_reached
        pause_disabled = replay_active or (not running) or max_epochs_reached
        prev_stage_disabled = replay_active or (not running) or safe_mode != 'semiauto' or int(training_state.get('stage_index', 0)) <= 0 or max_epochs_reached
        step_disabled = replay_active or (not running) or safe_mode != 'semiauto' or max_epochs_reached
        interval_disabled = replay_active or (not running) or safe_mode != 'auto' or max_epochs_reached
        interval_ms = SPEED_TO_INTERVAL[safe_play_speed]

        return params, training_state, button_label, start_disabled, pause_disabled, prev_stage_disabled, step_disabled, interval_disabled, interval_ms

    @app.callback(
        Output('level2-network-diagram-graph', 'figure'),
        Output('level2-math-explanation', 'children'),
        Output('level2-output-summary', 'children'),
        Output('level2-decision-boundary-graph', 'figure'),
        Output('level2-boundary-explanation', 'children'),
        Output('level2-activation-graph', 'figure'),
        Output('level2-model-status', 'children'),
        Output('level2-epoch-live', 'children'),
        Output('level2-training-stage-panel', 'children'),
        Input('level2-params-store', 'data'),
        Input('level2-training-store', 'data'),
        Input('level2-replay-store', 'data'),
    )
    def update_level2_views(params, training_store, replay_store):
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
        replay_active = bool((replay_store or {}).get('active'))
        replay_model_name = (replay_store or {}).get('model_name') or params.get('saved_model_name')
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
        explanation = make_level2_summary_panel(
            params,
            activation,
            dataset_bundle=dataset_bundle,
            probe_index=probe_index,
            current_stage=current_stage,
            pending_loss=pending_loss,
            pending_gradients=pending_gradients,
            gradient_norms=gradient_norms,
            learning_rate=learning_rate,
            previous_weights=previous_weights,
        )
        output_summary = make_level2_output_panel(metrics, dataset, params.get('history', {}))
        fig_boundary = make_decision_boundary_figure(dataset_bundle, params, activation)
        boundary_explanation = make_level2_boundary_explanation(dataset, activation_snapshot)
        fig_activation = make_activation_figure(activation)

        running = bool(training_state.get('running'))
        play_speed = training_state.get('play_speed', 'normal')
        if replay_active:
            model_status = 'Replay'
            status_color = '#7c3aed'
        elif running:
            model_status = 'Training'
            status_color = '#0f766e'
        elif metrics['epoch'] == 0 and current_stage == 'idle':
            model_status = 'Ready'
            status_color = '#1d4ed8'
        else:
            model_status = 'Idle'
            status_color = '#b45309'

        model_status_panel = [
            html.Div('Model Status', style={'fontSize': '11px', 'textTransform': 'uppercase', 'letterSpacing': '0.08em', 'color': '#64748b', 'marginBottom': '4px'}),
            html.Div(model_status, style={'fontSize': '21px', 'fontWeight': '700', 'color': status_color, 'marginBottom': '2px'}),
            html.Div(
                (
                    f'Replaying saved run {replay_model_name}.'
                    if replay_active and replay_model_name else 'Replaying a saved run.'
                ) if replay_active else (
                    'Training is active.' if running else (
                    'Model initialised.' if model_status == 'Ready' else f'Holding at {current_stage}.'
                    )
                ),
                style={'fontSize': '11px', 'color': '#64748b', 'lineHeight': '1.35'},
            ),
        ]

        epoch_panel = [
            html.Div('Epoch Count', style={'fontSize': '11px', 'textTransform': 'uppercase', 'letterSpacing': '0.08em', 'color': '#64748b', 'marginBottom': '4px'}),
            html.Div(str(metrics['epoch']), style={'fontSize': '24px', 'fontWeight': '700', 'color': '#0f172a', 'lineHeight': '1.05'}),
            html.Div(f'Current timeline epoch {epoch_in_progress}', style={'fontSize': '11px', 'color': '#64748b', 'marginTop': '3px'}),
        ]

        stage_panel = make_level2_training_stage_panel(
            current_stage=current_stage,
            epoch=metrics['epoch'],
            stage_details=None,
        )

        return (
            fig_network,
            explanation,
            output_summary,
            fig_boundary,
            boundary_explanation,
            fig_activation,
            model_status_panel,
            epoch_panel,
            stage_panel,
        )

    @app.callback(
        Output('level2-save-model-btn', 'disabled'),
        Output('level2-replay-saved-btn', 'disabled'),
        Output('level2-stop-replay-btn', 'disabled'),
        Input('level2-params-store', 'data'),
        Input('level2-replay-store', 'data'),
    )
    def update_level2_persistence_controls(params, replay_store):
        replay_active = bool((replay_store or {}).get('active'))
        replay_frames = (params or {}).get('replay_frames', [])
        current_epoch = int((params or {}).get('epoch', 0))

        save_disabled = replay_active or current_epoch <= 0
        replay_disabled = replay_active or len(replay_frames) <= 1
        stop_disabled = not replay_active
        return save_disabled, replay_disabled, stop_disabled

    @app.callback(
        Output('level2-save-feedback', 'children'),
        Output('model-history-refresh-store', 'data', allow_duplicate=True),
        Input('level2-save-model-btn', 'n_clicks'),
        State('level2-save-model-name', 'value'),
        State('level2-params-store', 'data'),
        State('user-session-store', 'data'),
        State('model-history-refresh-store', 'data'),
        prevent_initial_call=True,
    )
    def save_level2_model(n_clicks, requested_name, params, user_store, refresh_store):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        if not params or int(params.get('epoch', 0)) <= 0:
            return 'Train the Level 2 model before saving it to history.', dash.no_update

        learner_id = (user_store or {}).get('learner_id')
        if not learner_id:
            return 'User identity is not ready yet. Reload the page and try again.', dash.no_update

        meta = params.get('meta', {})
        dataset_bundle = build_level2_dataset(
            meta.get('dataset', 'moons'),
            input_dim=meta.get('input_dim', 2),
        )
        metrics = level2_evaluate_metrics(dataset_bundle, params, meta.get('activation', 'tanh'), l2=1e-4)
        model_name = (requested_name or '').strip() or f"Level 2 {meta.get('dataset', 'model')} epoch {params.get('epoch', 0)}"
        saved_run = save_level2_model_run(
            learner_id,
            model_name,
            _copy_params(params),
            metrics,
            display_name=(user_store or {}).get('display_name'),
        )
        refresh_count = int((refresh_store or {}).get('version', 0)) + 1
        return (
            f"Saved {saved_run['model_name']} to your Level 2 history.",
            {'version': refresh_count, 'last_saved_run_id': saved_run['id']},
        )

    @app.callback(
        Output('level2-params-store', 'data', allow_duplicate=True),
        Output('level2-training-store', 'data', allow_duplicate=True),
        Output('level2-replay-store', 'data', allow_duplicate=True),
        Output('level2-replay-interval', 'disabled', allow_duplicate=True),
        Output('level2-save-feedback', 'children', allow_duplicate=True),
        Output('level2-save-model-name', 'value', allow_duplicate=True),
        Input('url', 'pathname'),
        Input('url', 'search'),
        State('user-session-store', 'data'),
        prevent_initial_call='initial_duplicate',
    )
    def load_saved_level2_model(pathname, search, user_store):
        if pathname != '/level2':
            raise dash.exceptions.PreventUpdate

        query = parse_qs((search or '').lstrip('?'))
        run_id = query.get('model_run', [None])[0]
        if not run_id:
            raise dash.exceptions.PreventUpdate

        learner_id = (user_store or {}).get('learner_id')
        if not learner_id:
            return dash.no_update, dash.no_update, dash.no_update, True, 'User identity is not ready yet.', dash.no_update

        saved_run = get_level2_model_run(learner_id, int(run_id))
        if saved_run is None:
            return dash.no_update, dash.no_update, dash.no_update, True, 'Saved model not found for this user.', dash.no_update

        saved_params = _copy_params(saved_run.get('params', {}))
        replay_frames = _copy_replay_frames(saved_params.get('replay_frames', []))
        if not replay_frames:
            replay_frames = [_build_params_from_replay_frame(saved_params, [], saved_run_id=saved_run['id'], saved_model_name=saved_run['model_name'])]
            initial_params = replay_frames[0]
        else:
            initial_params = _build_params_from_replay_frame(
                replay_frames[0],
                replay_frames,
                saved_run_id=saved_run['id'],
                saved_model_name=saved_run['model_name'],
            )

        training_state = _make_training_state()
        training_state['current_stage'] = 'idle'
        replay_state = {
            'active': len(replay_frames) > 1,
            'current_index': 0,
            'loaded_run_id': saved_run['id'],
            'model_name': saved_run['model_name'],
        }
        replay_disabled = len(replay_frames) <= 1
        feedback = f"Loaded saved model {saved_run['model_name']} from your history."
        return initial_params, training_state, replay_state, replay_disabled, feedback, saved_run['model_name']

    @app.callback(
        Output('level2-params-store', 'data', allow_duplicate=True),
        Output('level2-replay-store', 'data', allow_duplicate=True),
        Output('level2-replay-interval', 'disabled', allow_duplicate=True),
        Input('level2-replay-saved-btn', 'n_clicks'),
        Input('level2-stop-replay-btn', 'n_clicks'),
        Input('level2-replay-interval', 'n_intervals'),
        State('level2-params-store', 'data'),
        State('level2-replay-store', 'data'),
        prevent_initial_call=True,
    )
    def replay_saved_level2_model(n_replay, n_stop, n_intervals, params, replay_store):
        _ = (n_replay, n_stop, n_intervals)
        if not params:
            raise dash.exceptions.PreventUpdate

        replay_frames = _copy_replay_frames(params.get('replay_frames', []))
        if len(replay_frames) <= 1:
            raise dash.exceptions.PreventUpdate

        current_replay = dict(replay_store or {})
        trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0] if dash.callback_context.triggered else None
        model_name = current_replay.get('model_name') or params.get('saved_model_name')
        run_id = current_replay.get('loaded_run_id') or params.get('saved_run_id')

        if trigger == 'level2-stop-replay-btn':
            current_replay['active'] = False
            return dash.no_update, current_replay, True

        if trigger == 'level2-replay-saved-btn':
            current_replay = {
                'active': True,
                'current_index': 0,
                'loaded_run_id': run_id,
                'model_name': model_name,
            }
            return _build_params_from_replay_frame(replay_frames[0], replay_frames, saved_run_id=run_id, saved_model_name=model_name), current_replay, False

        if trigger != 'level2-replay-interval' or not current_replay.get('active'):
            raise dash.exceptions.PreventUpdate

        next_index = int(current_replay.get('current_index', 0)) + 1
        if next_index >= len(replay_frames):
            final_index = len(replay_frames) - 1
            current_replay['active'] = False
            current_replay['current_index'] = final_index
            return _build_params_from_replay_frame(replay_frames[final_index], replay_frames, saved_run_id=run_id, saved_model_name=model_name), current_replay, True

        current_replay['current_index'] = next_index
        return _build_params_from_replay_frame(replay_frames[next_index], replay_frames, saved_run_id=run_id, saved_model_name=model_name), current_replay, False