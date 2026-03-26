from importlib import import_module

import dash

try:
	from dash import Input, Output, State, html
except ImportError:
	dash_dependencies = import_module('dash.dependencies')
	Input = dash_dependencies.Input
	Output = dash_dependencies.Output
	State = dash_dependencies.State
	from dash import html

from pySecProgramming.code_execution import execute_python_snippet
from pages.levels.level2 import (
	level2_evaluate_metrics,
	make_decision_boundary_figure,
	make_level2_output_panel,
	make_level2_summary_panel,
	make_level2_training_curves_figure,
	make_network_diagram_figure,
)
from pages.levels.level2.methods import (
	apply_level2_gradients,
	compute_level2_gradients,
	make_level2_activation_snapshot,
	make_level2_training_stage_panel,
)
from pages.levels.level3.methods import (
	build_level3_execution_environment,
	level3_activation_heatmap_figure,
	level3_build_dataset,
	level3_dataset_preview_figure,
	level3_dataset_summary_children,
	level3_execution_live_children,
	level3_extract_meta_from_code,
	level3_initialise_model,
	level3_initialise_store,
	level3_model_matches,
	level3_model_status_children,
	level3_notebook_status_children,
	level3_setup_complete,
	level3_training_log_children,
	make_level3_placeholder_figure,
)
from pages.levels.level3.layout import LEVEL3_CELL_EDITORS


SPEED_TO_INTERVAL = {
	'slow': 700,
	'normal': 350,
	'fast': 180,
}
STAGE_SEQUENCE = ('forward', 'loss', 'backward', 'update')


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


def _make_training_state(max_epochs, play_speed='normal', mode='auto'):
	safe_mode = mode if mode in {'auto', 'semiauto'} else 'auto'
	safe_speed = play_speed if play_speed in SPEED_TO_INTERVAL else 'normal'
	return {
		'running': False,
		'paused': False,
		'max_epochs': int(max_epochs),
		'probe_index': 0,
		'previous_weights': None,
		'play_speed': safe_speed,
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


def _append_training_log(store, dataset_bundle, meta):
	model = store.get('model')
	if model is None or int(model.get('epoch', 0)) <= 0:
		return store

	metrics = level2_evaluate_metrics(dataset_bundle, model, meta['activation'], l2=1e-4)
	training_logs = list(store.get('training_logs', []))
	epoch = int(model.get('epoch', 0))
	entry = {
		'run_number': len(training_logs) + 1,
		'epoch': epoch,
		'epochs': meta['epochs'],
		'learning_rate': meta['learning_rate'],
		'train_loss': metrics['train_loss'],
		'train_accuracy': metrics['train_accuracy'],
		'test_loss': metrics['test_loss'],
		'test_accuracy': metrics['test_accuracy'],
	}

	if training_logs and training_logs[-1].get('epoch') == epoch:
		training_logs[-1].update(entry)
		training_logs[-1]['run_number'] = training_logs[-1].get('run_number', len(training_logs))
	else:
		training_logs.append(entry)

	store['training_logs'] = training_logs
	return store


def register_level3_callbacks(app):
	@app.callback(
		Output('level3-params-store', 'data'),
		Output('level3-training-store', 'data'),
		Output('level3-train-toggle-btn', 'children'),
		Output('level3-train-toggle-btn', 'disabled'),
		Output('level3-pause-btn', 'disabled'),
		Output('level3-prev-stage-btn', 'disabled'),
		Output('level3-step-btn', 'disabled'),
		Output('level3-train-interval', 'disabled'),
		Output('level3-train-interval', 'interval'),
		Input('level3-load-data-btn', 'n_clicks'),
		Input('level3-input-layer-btn', 'n_clicks'),
		Input('level3-activation-btn', 'n_clicks'),
		Input('level3-hidden-layers-btn', 'n_clicks'),
		Input('level3-output-layer-btn', 'n_clicks'),
		Input('level3-training-config-btn', 'n_clicks'),
		Input('level3-train-toggle-btn', 'n_clicks'),
		Input('level3-pause-btn', 'n_clicks'),
		Input('level3-prev-stage-btn', 'n_clicks'),
		Input('level3-step-btn', 'n_clicks'),
		Input('level3-reset-btn', 'n_clicks'),
		Input('level3-train-interval', 'n_intervals'),
		Input('level3-play-speed-control', 'value'),
		Input('level3-training-mode', 'value'),
		State('level3-cell-1-code', 'value'),
		State('level3-cell-2-code', 'value'),
		State('level3-cell-3-code', 'value'),
		State('level3-cell-4-code', 'value'),
		State('level3-cell-5-code', 'value'),
		State('level3-cell-6-code', 'value'),
		State('level3-params-store', 'data'),
		State('level3-training-store', 'data'),
	)
	def update_level3_state(
		n_load,
		n_input,
		n_activation,
		n_hidden,
		n_output,
		n_training_config,
		n_train_toggle,
		n_pause,
		n_prev_stage,
		n_step,
		n_reset,
		n_intervals,
		play_speed,
		training_mode,
		cell_1_code,
		cell_2_code,
		cell_3_code,
		cell_4_code,
		cell_5_code,
		cell_6_code,
		store,
		training_store,
	):
		_ = (
			n_load,
			n_input,
			n_activation,
			n_hidden,
			n_output,
			n_training_config,
			n_train_toggle,
			n_pause,
			n_prev_stage,
			n_step,
			n_reset,
			n_intervals,
		)
		ctx = dash.callback_context
		trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

		button_to_key = {
			'level3-load-data-btn': 'load_dataset',
			'level3-input-layer-btn': 'input_layer',
			'level3-activation-btn': 'activation',
			'level3-hidden-layers-btn': 'hidden_layers',
			'level3-output-layer-btn': 'output_layer',
			'level3-training-config-btn': 'training_config',
		}
		code_triggers = set(button_to_key)
		extracted_meta = level3_extract_meta_from_code(
			cell_1_code,
			cell_2_code,
			cell_3_code,
			cell_4_code,
			cell_5_code,
			cell_6_code,
		)

		if store is None:
			meta = extracted_meta
			store = level3_initialise_store(meta)
		else:
			meta = extracted_meta if trigger in code_triggers or trigger == 'level3-reset-btn' else store.get('meta', extracted_meta)
			store['meta'] = meta

		safe_play_speed = play_speed if play_speed in SPEED_TO_INTERVAL else 'normal'
		safe_mode = training_mode if training_mode in {'auto', 'semiauto'} else 'auto'
		training_state = training_store or _make_training_state(meta['epochs'], play_speed=safe_play_speed, mode=safe_mode)
		training_state['play_speed'] = safe_play_speed
		training_state['mode'] = safe_mode
		training_state['max_epochs'] = int(meta['epochs'])

		running = bool(training_state.get('running'))
		paused = bool(training_state.get('paused'))

		if trigger in code_triggers:
			previous_runs = dict(store.get('cell_runs', {}))
			store = level3_initialise_store(meta)
			for key, value in previous_runs.items():
				if key in store['cell_runs']:
					store['cell_runs'][key] = value
				store['cell_runs'][button_to_key[trigger]] = True
			store = level3_initialise_model(store, meta)
			training_state = _make_training_state(meta['epochs'], play_speed=safe_play_speed, mode=safe_mode)
			running = False
			paused = False

		elif trigger == 'level3-reset-btn':
			previous_runs = dict(store.get('cell_runs', {}))
			store = level3_initialise_store(meta)
			for key, value in previous_runs.items():
				if key in store['cell_runs']:
					store['cell_runs'][key] = value
			if any(store['cell_runs'].values()):
				store = level3_initialise_model(store, meta)
			training_state = _make_training_state(meta['epochs'], play_speed=safe_play_speed, mode=safe_mode)
			running = False
			paused = False

		setup_complete = level3_setup_complete(store)
		dataset_bundle = level3_build_dataset(meta)

		if trigger == 'level3-train-toggle-btn' and setup_complete:
			if store.get('model') is None or not level3_model_matches(store, meta):
				store = level3_initialise_model(store, meta)
			if store['model'].get('epoch', 0) < meta['epochs']:
				running = True
				paused = False
				training_state = _ensure_epoch_cycle(
					training_state,
					dataset_bundle,
					store['model'],
					meta['activation'],
					meta['learning_rate'],
					meta,
				)
				if training_state.get('current_stage') == 'idle':
					training_state['stage_index'] = 0
				store['model'], training_state = _apply_stage_view(training_state, meta)

		elif trigger == 'level3-pause-btn' and running:
			running = False
			paused = True

		elif trigger in {'level3-train-interval', 'level3-step-btn', 'level3-prev-stage-btn'} and running and setup_complete:
			allow_auto_advance = trigger == 'level3-train-interval' and safe_mode == 'auto'
			allow_next = trigger == 'level3-step-btn' and safe_mode == 'semiauto'
			allow_prev = trigger == 'level3-prev-stage-btn' and safe_mode == 'semiauto'

			if allow_auto_advance or allow_next or allow_prev:
				training_state = _ensure_epoch_cycle(
					training_state,
					dataset_bundle,
					store['model'],
					meta['activation'],
					meta['learning_rate'],
					meta,
				)
				stage_index = int(training_state.get('stage_index', 0))

				if allow_prev:
					training_state['stage_index'] = max(0, stage_index - 1)
					store['model'], training_state = _apply_stage_view(training_state, meta)
				else:
					if stage_index < len(STAGE_SEQUENCE) - 1:
						training_state['stage_index'] = stage_index + 1
						store['model'], training_state = _apply_stage_view(training_state, meta)
					else:
						store['model'] = _copy_params(training_state.get('epoch_updated_params')) or store['model']
						store['model']['meta'] = dict(meta)
						store = _append_training_log(store, dataset_bundle, meta)

						if store['model'].get('epoch', 0) >= meta['epochs']:
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
								store['model'],
								meta['activation'],
								meta['learning_rate'],
								meta,
							)
							training_state['stage_index'] = 0
							store['model'], training_state = _apply_stage_view(training_state, meta)

		if store.get('model') is not None:
			store['model']['meta'] = dict(meta)

		max_epochs_reached = bool(store.get('model')) and store['model'].get('epoch', 0) >= meta['epochs']
		if max_epochs_reached:
			running = False
			paused = False
			training_state = _clear_epoch_cycle(training_state)
			training_state['stage_index'] = 0
			training_state['current_stage'] = 'idle'

		if not running and not paused and store.get('model') is not None and store['model'].get('epoch', 0) == 0 and int(training_state.get('stage_index', 0)) == 0:
			training_state['current_stage'] = 'idle'

		training_state = {
			'running': running,
			'paused': paused,
			'max_epochs': int(meta['epochs']),
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

		setup_complete = level3_setup_complete(store)
		if not setup_complete:
			button_label = 'Finish Setup Cells'
		elif max_epochs_reached:
			button_label = 'Max Epochs Reached'
		elif running:
			button_label = 'Training Active'
		elif paused or (store.get('model') is not None and (store['model'].get('epoch', 0) > 0 or training_state.get('current_stage') != 'idle')):
			button_label = 'Resume Training'
		else:
			button_label = 'Start Training'

		start_disabled = (not setup_complete) or running or max_epochs_reached or store.get('model') is None
		pause_disabled = (not running) or max_epochs_reached
		prev_stage_disabled = (not running) or safe_mode != 'semiauto' or int(training_state.get('stage_index', 0)) <= 0 or max_epochs_reached or (not setup_complete)
		step_disabled = (not running) or safe_mode != 'semiauto' or max_epochs_reached or (not setup_complete)
		interval_disabled = (not running) or safe_mode != 'auto' or max_epochs_reached or (not setup_complete)
		interval_ms = SPEED_TO_INTERVAL[safe_play_speed]

		return (
			store,
			training_state,
			button_label,
			start_disabled,
			pause_disabled,
			prev_stage_disabled,
			step_disabled,
			interval_disabled,
			interval_ms,
		)

	@app.callback(
		Output('level3-dataset-preview-graph', 'figure'),
		Output('level3-dataset-summary', 'children'),
		Output('level3-network-diagram-graph', 'figure'),
		Output('level3-arch-summary', 'children'),
		Output('level3-boundary-graph', 'figure'),
		Output('level3-loss-graph', 'figure'),
		Output('level3-activations-graph', 'figure'),
		Output('level3-training-log', 'children'),
		Output('level3-notebook-status', 'children'),
		Output('level3-model-status', 'children'),
		Output('level3-execution-live', 'children'),
		Output('level3-output-summary', 'children'),
		Output('level3-training-stage-panel', 'children'),
		Input('level3-params-store', 'data'),
		Input('level3-training-store', 'data'),
	)
	def update_level3_views(store, training_state):
		dataset_placeholder = make_level3_placeholder_figure(
			'Dataset preview',
			'Run Cell 1 to load a toy classification dataset.',
		)
		model_placeholder = make_level3_placeholder_figure(
			'Architecture diagram',
			'Run the structure cells to define the feed-forward network.',
		)
		boundary_placeholder = make_level3_placeholder_figure(
			'Decision boundary / prediction surface',
			'Complete the setup cells to render the model boundary.',
		)
		loss_placeholder = make_level3_placeholder_figure(
			'Training curves',
			'Start cached training to populate optimisation history.',
		)
		activation_placeholder = make_level3_placeholder_figure(
			'Per-layer activations',
			'The activation snapshot appears after the structure has been committed.',
		)

		if store is None:
			return (
				dataset_placeholder,
				html.Div('Run Cell 1 to create the dataset preview.', style={'color': '#64748b'}),
				model_placeholder,
				html.Div('Run the structure cells to generate the architecture summary.', style={'color': '#64748b'}),
				boundary_placeholder,
				loss_placeholder,
				activation_placeholder,
				level3_training_log_children([]),
				level3_notebook_status_children(None),
				level3_model_status_children(None),
				level3_execution_live_children(None),
				html.Div('Finish the setup cells to populate the metrics panel.', style={'color': '#64748b', 'fontSize': '12px'}),
				make_level2_training_stage_panel(current_stage='idle', epoch=0),
			)

		meta = store['meta']
		dataset_bundle = level3_build_dataset(meta)
		model = store.get('model')
		training_state = training_state or {}
		current_stage = training_state.get('current_stage', 'idle')
		probe_index = int(training_state.get('probe_index', 0))

		dataset_preview = level3_dataset_preview_figure(dataset_bundle)
		dataset_summary = level3_dataset_summary_children(dataset_bundle, meta)
		network_fig = model_placeholder
		arch_summary = html.Div('Run the structure cells to generate the architecture summary.', style={'color': '#64748b'})
		boundary_fig = boundary_placeholder
		loss_fig = loss_placeholder
		activations_fig = activation_placeholder
		output_summary = html.Div('Finish the setup cells to populate the metrics panel.', style={'color': '#64748b', 'fontSize': '12px'})

		if model is not None:
			history = model.get('history', {})
			activation_snapshot = make_level2_activation_snapshot(
				dataset_bundle,
				model,
				meta['activation'],
				probe_index=probe_index,
			)
			network_fig = make_network_diagram_figure(
				input_dim=meta['input_dim'],
				hidden_layers=meta['hidden_layer_sizes'],
				output_dim=meta['output_dim'],
				params=model,
				activation=meta['activation'],
				feature_names=dataset_bundle['feature_names'],
				activation_snapshot=activation_snapshot,
				previous_weights=training_state.get('previous_weights'),
				is_training=current_stage != 'idle',
				stage_name=current_stage,
				gradient_snapshot=training_state.get('pending_gradients'),
			)
			arch_summary = make_level2_summary_panel(
				model,
				meta['activation'],
				dataset_bundle=dataset_bundle,
				probe_index=probe_index,
				current_stage=current_stage,
				pending_loss=training_state.get('pending_loss'),
				pending_gradients=training_state.get('pending_gradients'),
				gradient_norms=training_state.get('gradient_norms'),
				learning_rate=meta['learning_rate'],
				previous_weights=training_state.get('previous_weights'),
			)
			boundary_fig = make_decision_boundary_figure(dataset_bundle, model, meta['activation'])
			if current_stage != 'idle':
				boundary_fig.update_layout(title=f"Cached training view - epoch {model.get('epoch', 0) + 1} ({current_stage.title()})")
			elif model.get('epoch', 0) > 0:
				boundary_fig.update_layout(title=f"Decision boundary after {model.get('epoch', 0)} cached epochs")
			else:
				boundary_fig.update_layout(title='Initial decision surface from the committed architecture')

			if history.get('train_loss'):
				loss_fig = make_level2_training_curves_figure(history)
				loss_fig.update_layout(title='Cached training loss and accuracy')

			activations_fig = level3_activation_heatmap_figure(model, dataset_bundle['X_test'], meta['activation'])
			output_summary = make_level2_output_panel(
				level2_evaluate_metrics(dataset_bundle, model, meta['activation'], l2=1e-4),
				meta['dataset'],
				history,
			)

		return (
			dataset_preview,
			dataset_summary,
			network_fig,
			arch_summary,
			boundary_fig,
			loss_fig,
			activations_fig,
			level3_training_log_children(store.get('training_logs', [])),
			level3_notebook_status_children(store),
			level3_model_status_children(store),
			level3_execution_live_children(store),
			output_summary,
			make_level2_training_stage_panel(current_stage=current_stage, epoch=model.get('epoch', 0) if model else 0),
		)

	@app.callback(
		Output('level3-cell-1-console', 'children'),
		Output('level3-cell-2-console', 'children'),
		Output('level3-cell-3-console', 'children'),
		Output('level3-cell-4-console', 'children'),
		Output('level3-cell-5-console', 'children'),
		Output('level3-cell-6-console', 'children'),
		Output('level3-dataset-output-box', 'children'),
		Input('level3-load-data-btn', 'n_clicks'),
		Input('level3-input-layer-btn', 'n_clicks'),
		Input('level3-activation-btn', 'n_clicks'),
		Input('level3-hidden-layers-btn', 'n_clicks'),
		Input('level3-output-layer-btn', 'n_clicks'),
		Input('level3-training-config-btn', 'n_clicks'),
		State('level3-cell-1-code', 'value'),
		State('level3-cell-2-code', 'value'),
		State('level3-cell-3-code', 'value'),
		State('level3-cell-4-code', 'value'),
		State('level3-cell-5-code', 'value'),
		State('level3-cell-6-code', 'value'),
		prevent_initial_call=True,
	)
	def execute_level3_cells(
		n_load,
		n_input,
		n_activation,
		n_hidden,
		n_output,
		n_training_config,
		cell_1_code,
		cell_2_code,
		cell_3_code,
		cell_4_code,
		cell_5_code,
		cell_6_code,
	):
		_ = (n_load, n_input, n_activation, n_hidden, n_output, n_training_config)
		ctx = dash.callback_context
		if not ctx.triggered:
			raise dash.exceptions.PreventUpdate

		trigger = ctx.triggered[0]['prop_id'].split('.')[0]
		button_to_cell = {
			'level3-load-data-btn': (1, cell_1_code),
			'level3-input-layer-btn': (2, cell_2_code),
			'level3-activation-btn': (3, cell_3_code),
			'level3-hidden-layers-btn': (4, cell_4_code),
			'level3-output-layer-btn': (5, cell_5_code),
			'level3-training-config-btn': (6, cell_6_code),
		}
		cell_number, code = button_to_cell[trigger]
		meta = level3_extract_meta_from_code(
			cell_1_code,
			cell_2_code,
			cell_3_code,
			cell_4_code,
			cell_5_code,
			cell_6_code,
		)
		execution_env = build_level3_execution_environment(cell_number, meta)
		output_children, error_children, _ = execute_python_snippet(code, execution_env)
		console_children = error_children if error_children else output_children

		responses = [dash.no_update] * 7
		responses[cell_number - 1] = console_children
		if cell_number == 1:
			responses[6] = console_children
		return tuple(responses)

	@app.callback(
		Output('level3-cell-1-validation', 'children'),
		Output('level3-cell-1-highlighted', 'children'),
		Output('level3-cell-2-validation', 'children'),
		Output('level3-cell-2-highlighted', 'children'),
		Output('level3-cell-3-validation', 'children'),
		Output('level3-cell-3-highlighted', 'children'),
		Output('level3-cell-4-validation', 'children'),
		Output('level3-cell-4-highlighted', 'children'),
		Output('level3-cell-5-validation', 'children'),
		Output('level3-cell-5-highlighted', 'children'),
		Output('level3-cell-6-validation', 'children'),
		Output('level3-cell-6-highlighted', 'children'),
		Input('level3-cell-1-code', 'value'),
		Input('level3-cell-2-code', 'value'),
		Input('level3-cell-3-code', 'value'),
		Input('level3-cell-4-code', 'value'),
		Input('level3-cell-5-code', 'value'),
		Input('level3-cell-6-code', 'value'),
	)
	def update_level3_code_feedback(
		cell_1_code,
		cell_2_code,
		cell_3_code,
		cell_4_code,
		cell_5_code,
		cell_6_code,
	):
		responses = []
		for cell_number, code in enumerate(
			[cell_1_code, cell_2_code, cell_3_code, cell_4_code, cell_5_code, cell_6_code],
			start=1,
		):
			editor = LEVEL3_CELL_EDITORS[cell_number]
			responses.append(editor.build_validation_message(code))
			responses.append(editor.build_highlighted_code(code))
		return tuple(responses)