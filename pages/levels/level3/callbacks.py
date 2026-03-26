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

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from pySecProgramming.code_execution import execute_python_snippet
from pages.levels.level2 import (
	level2_evaluate_metrics,
	level2_forward_pass,
	make_decision_boundary_figure,
	make_level2_output_panel,
	make_level2_summary_panel,
	make_level2_training_curves_figure,
	make_network_diagram_figure,
	train_level2_model,
)
from pages.levels.level3.methods import (
	build_level3_execution_environment,
	level3_activation_heatmap_figure,
	level3_build_dataset,
	level3_confusion_matrix_figure,
	level3_dataset_preview_figure,
	level3_dataset_summary_children,
	level3_execution_live_children,
	level3_extract_meta_from_code,
	level3_forward_summary_children,
	level3_hidden_space_figure,
	level3_initialise_model,
	level3_initialise_store,
	level3_metrics_summary_children,
	level3_misclassified_figure,
	level3_model_matches,
	level3_model_status_children,
	level3_notebook_status_children,
	level3_training_log_children,
	make_level3_placeholder_figure,
)
from pages.levels.level3.layout import LEVEL3_CELL_EDITORS


def register_level3_callbacks(app):
	@app.callback(
		Output('level3-params-store', 'data'),
		Input('level3-load-data-btn', 'n_clicks'),
		Input('level3-define-model-btn', 'n_clicks'),
		Input('level3-forward-btn', 'n_clicks'),
		Input('level3-train-btn', 'n_clicks'),
		Input('level3-inspect-btn', 'n_clicks'),
		Input('level3-evaluate-btn', 'n_clicks'),
		State('level3-cell-1-code', 'value'),
		State('level3-cell-2-code', 'value'),
		State('level3-cell-4-code', 'value'),
		State('level3-params-store', 'data'),
	)
	def update_level3_params(
		n_load,
		n_define,
		n_forward,
		n_train,
		n_inspect,
		n_evaluate,
		cell_1_code,
		cell_2_code,
		cell_4_code,
		store,
	):
		_ = (n_load, n_define, n_forward, n_train, n_inspect, n_evaluate)
		ctx = dash.callback_context
		trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
		meta = level3_extract_meta_from_code(cell_1_code, cell_2_code, cell_4_code)

		if store is None or store.get('meta') != meta:
			store = level3_initialise_store(meta)
		else:
			store['meta'] = meta

		if trigger in (None, 'level3-load-data-btn'):
			store = level3_initialise_store(meta)
			store['cell_runs']['load_dataset'] = True
			return store

		store['cell_runs']['load_dataset'] = True

		if trigger == 'level3-define-model-btn':
			store = level3_initialise_model(store, meta)
			store['cell_runs']['define_model'] = True
			return store

		if store.get('model') is None or not level3_model_matches(store, meta):
			store = level3_initialise_model(store, meta)
		store['cell_runs']['define_model'] = True

		dataset_bundle = level3_build_dataset(meta)

		if trigger == 'level3-forward-btn':
			batch_X = dataset_bundle['X_train'][:24]
			batch_y = dataset_bundle['y_train'][:24]
			predictions, cache = level2_forward_pass(batch_X, store['model'], meta['activation'])
			store['forward_summary'] = {
				'batch_shape': list(batch_X.shape),
				'hidden_shapes': [list(activation_matrix.shape) for activation_matrix in cache['activations'][1:-1]],
				'output_shape': list(predictions.shape),
				'examples': [
					{
						'x1': float(dataset_bundle['X_train_raw'][index, 0]),
						'x2': float(dataset_bundle['X_train_raw'][index, 1]),
						'target': int(batch_y[index]),
						'probability': float(predictions[0, index]) if predictions.shape[0] == 1 else float(np.max(predictions[:, index])),
					}
					for index in range(min(5, batch_X.shape[0]))
				],
			}
			store['cell_runs']['forward_pass'] = True

		elif trigger == 'level3-train-btn':
			store['model'] = train_level2_model(
				dataset_bundle,
				store['model'],
				activation=meta['activation'],
				epochs=meta['epochs'],
				lr=meta['learning_rate'],
				l2=1e-4,
			)
			metrics = level2_evaluate_metrics(dataset_bundle, store['model'], meta['activation'], l2=1e-4)
			store['training_logs'].append({
				'run_number': len(store['training_logs']) + 1,
				'epochs': meta['epochs'],
				'learning_rate': meta['learning_rate'],
				'train_loss': metrics['train_loss'],
				'train_accuracy': metrics['train_accuracy'],
				'test_loss': metrics['test_loss'],
				'test_accuracy': metrics['test_accuracy'],
			})
			store['cell_runs']['train_model'] = True

		elif trigger == 'level3-inspect-btn':
			store['inspect_ran'] = True
			store['cell_runs']['inspect'] = True

		elif trigger == 'level3-evaluate-btn':
			predictions, _ = level2_forward_pass(dataset_bundle['X_test'], store['model'], meta['activation'])
			if predictions.shape[0] == 1:
				pred_labels = (predictions.flatten() >= 0.5).astype(np.int32)
			else:
				pred_labels = np.argmax(predictions, axis=0).astype(np.int32)
			confusion_values = confusion_matrix(dataset_bundle['y_test'], pred_labels, labels=[0, 1])
			precision, recall, f1, support = precision_recall_fscore_support(
				dataset_bundle['y_test'],
				pred_labels,
				labels=[0, 1],
				zero_division=0,
			)
			store['evaluation'] = {
				'metrics': level2_evaluate_metrics(dataset_bundle, store['model'], meta['activation'], l2=1e-4),
				'confusion_matrix': confusion_values.tolist(),
				'pred_labels': pred_labels.tolist(),
				'precision': precision.tolist(),
				'recall': recall.tolist(),
				'f1': f1.tolist(),
				'support': support.tolist(),
				'misclassified_count': int(np.sum(pred_labels != dataset_bundle['y_test'])),
				'sample_count': int(dataset_bundle['y_test'].shape[0]),
			}
			store['cell_runs']['evaluate'] = True

		return store

	@app.callback(
		Output('level3-boundary-graph', 'figure'),
		Output('level3-loss-graph', 'figure'),
		Output('level3-activations-graph', 'figure'),
		Output('level3-dataset-preview-graph', 'figure'),
		Output('level3-dataset-summary', 'children'),
		Output('level3-network-diagram-graph', 'figure'),
		Output('level3-arch-summary', 'children'),
		Output('level3-forward-output', 'children'),
		Output('level3-training-log', 'children'),
		Output('level3-hidden-space-graph', 'figure'),
		Output('level3-confusion-matrix-graph', 'figure'),
		Output('level3-misclassified-graph', 'figure'),
		Output('level3-metrics-summary', 'children'),
		Output('level3-notebook-status', 'children'),
		Output('level3-model-status', 'children'),
		Output('level3-execution-live', 'children'),
		Output('level3-output-summary', 'children'),
		Input('level3-params-store', 'data'),
	)
	def update_level3_views(store):
		boundary_placeholder = make_level3_placeholder_figure(
			'Decision boundary / prediction surface',
			'Run Cell 2 to define a classifier and render its decision surface.',
		)
		loss_placeholder = make_level3_placeholder_figure(
			'Training curves',
			'Run Cell 4 to train the model and record optimisation history.',
		)
		activation_placeholder = make_level3_placeholder_figure(
			'Per-layer activations',
			'Run Cell 5 to inspect hidden activations after training.',
		)
		dataset_placeholder = make_level3_placeholder_figure(
			'Dataset preview',
			'Run Cell 1 to load a toy classification dataset.',
		)
		model_placeholder = make_level3_placeholder_figure(
			'Architecture diagram',
			'Run Cell 2 to define the hidden stack and output layer.',
		)
		hidden_placeholder = make_level3_placeholder_figure(
			'Hidden-space projection',
			'Run Cell 5 to inspect hidden representations.',
		)
		eval_placeholder = make_level3_placeholder_figure(
			'Evaluation',
			'Run Cell 6 to compute confusion and evaluation diagnostics.',
		)

		if store is None:
			return (
				boundary_placeholder,
				loss_placeholder,
				activation_placeholder,
				dataset_placeholder,
				html.Div('Run Cell 1 to create the dataset preview.', style={'color': '#64748b'}),
				model_placeholder,
				html.Div('Run Cell 2 to generate the architecture summary.', style={'color': '#64748b'}),
				level3_forward_summary_children(None),
				level3_training_log_children([]),
				hidden_placeholder,
				eval_placeholder,
				eval_placeholder,
				level3_metrics_summary_children(None),
				level3_notebook_status_children(None),
				level3_model_status_children(None),
				level3_execution_live_children(None),
				html.Div('Run Cell 4 to populate the metrics panel.', style={'color': '#64748b', 'fontSize': '12px'}),
			)

		meta = store['meta']
		dataset_bundle = level3_build_dataset(meta)
		dataset_preview = level3_dataset_preview_figure(dataset_bundle)
		dataset_summary = level3_dataset_summary_children(dataset_bundle, meta)
		notebook_status = level3_notebook_status_children(store)
		model_status = level3_model_status_children(store)
		execution_live = level3_execution_live_children(store)

		boundary_fig = boundary_placeholder
		loss_fig = loss_placeholder
		activations_fig = activation_placeholder
		network_fig = model_placeholder
		arch_summary = html.Div('Run Cell 2 to generate the architecture summary.', style={'color': '#64748b'})
		hidden_space_fig = hidden_placeholder
		confusion_fig = eval_placeholder
		misclassified_fig = eval_placeholder
		output_summary = html.Div('Run Cell 4 to populate the metrics panel.', style={'color': '#64748b', 'fontSize': '12px'})

		model = store.get('model')
		if model is not None:
			boundary_fig = make_decision_boundary_figure(dataset_bundle, model, meta['activation'])
			if store['cell_runs']['train_model']:
				boundary_fig.update_layout(title=f"Trained decision boundary after {model.get('epoch', 0)} epochs")
			else:
				boundary_fig.update_layout(title='Initial decision surface from the current model definition')

			history = model.get('history', {})
			if history.get('train_loss'):
				loss_fig = make_level2_training_curves_figure(history)
				loss_fig.update_layout(title='Training loss and accuracy from Cell 4')

			network_fig = make_network_diagram_figure(
				input_dim=meta['input_dim'],
				hidden_layers=meta['hidden_layer_sizes'],
				output_dim=meta['output_dim'],
				params=model,
				activation=meta['activation'],
				feature_names=dataset_bundle['feature_names'],
			)
			arch_summary = make_level2_summary_panel(model, meta['activation'], dataset_bundle=dataset_bundle)
			output_summary = make_level2_output_panel(
				level2_evaluate_metrics(dataset_bundle, model, meta['activation'], l2=1e-4),
				meta['dataset'],
				history,
			)

			if store['inspect_ran']:
				activations_fig = level3_activation_heatmap_figure(model, dataset_bundle['X_test'], meta['activation'])
				hidden_space_fig = level3_hidden_space_figure(model, dataset_bundle['X_test'], dataset_bundle['y_test'], meta['activation'])

		forward_output = level3_forward_summary_children(store.get('forward_summary'))
		training_log = level3_training_log_children(store.get('training_logs', []))

		evaluation = store.get('evaluation')
		if evaluation is not None:
			confusion_fig = level3_confusion_matrix_figure(evaluation['confusion_matrix'])
			pred_labels = np.array(evaluation['pred_labels'], dtype=np.int32)
			misclassified_fig = level3_misclassified_figure(dataset_bundle, pred_labels)
		metrics_summary = level3_metrics_summary_children(evaluation)

		return (
			boundary_fig,
			loss_fig,
			activations_fig,
			dataset_preview,
			dataset_summary,
			network_fig,
			arch_summary,
			forward_output,
			training_log,
			hidden_space_fig,
			confusion_fig,
			misclassified_fig,
			metrics_summary,
			notebook_status,
			model_status,
			execution_live,
			output_summary,
		)

	@app.callback(
		Output('level3-cell-1-console', 'children'),
		Output('level3-cell-2-console', 'children'),
		Output('level3-cell-3-console', 'children'),
		Output('level3-cell-4-console', 'children'),
		Output('level3-cell-5-console', 'children'),
		Output('level3-cell-6-console', 'children'),
		Input('level3-load-data-btn', 'n_clicks'),
		Input('level3-define-model-btn', 'n_clicks'),
		Input('level3-forward-btn', 'n_clicks'),
		Input('level3-train-btn', 'n_clicks'),
		Input('level3-inspect-btn', 'n_clicks'),
		Input('level3-evaluate-btn', 'n_clicks'),
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
		n_define,
		n_forward,
		n_train,
		n_inspect,
		n_evaluate,
		cell_1_code,
		cell_2_code,
		cell_3_code,
		cell_4_code,
		cell_5_code,
		cell_6_code,
	):
		_ = (n_load, n_define, n_forward, n_train, n_inspect, n_evaluate)
		ctx = dash.callback_context
		if not ctx.triggered:
			raise dash.exceptions.PreventUpdate

		trigger = ctx.triggered[0]['prop_id'].split('.')[0]
		button_to_cell = {
			'level3-load-data-btn': (1, cell_1_code),
			'level3-define-model-btn': (2, cell_2_code),
			'level3-forward-btn': (3, cell_3_code),
			'level3-train-btn': (4, cell_4_code),
			'level3-inspect-btn': (5, cell_5_code),
			'level3-evaluate-btn': (6, cell_6_code),
		}

		cell_number, code = button_to_cell[trigger]
		meta = level3_extract_meta_from_code(cell_1_code, cell_2_code, cell_4_code)
		execution_env = build_level3_execution_environment(cell_number, meta)
		output_children, error_children, _ = execute_python_snippet(code, execution_env)
		console_children = error_children if error_children else output_children

		responses = [dash.no_update] * 6
		responses[cell_number - 1] = console_children
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