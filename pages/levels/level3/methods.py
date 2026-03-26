import ast

import numpy as np
import plotly.graph_objects as go
from dash import html
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

from pages.levels.level2 import (
	build_level2_dataset,
	init_level2_mlp,
	level2_evaluate_metrics,
	level2_forward_pass,
	level2_set_baseline_history,
	train_level2_model,
)


DEFAULT_LEVEL3_META = {
	'dataset': 'moons',
	'input_dim': 2,
	'hidden_layer_sizes': [6, 6],
	'activation': 'tanh',
	'epochs': 120,
	'learning_rate': 0.08,
	'output_dim': 1,
}


def make_level3_placeholder_figure(title, message):
	fig = go.Figure()
	fig.add_annotation(
		text=message,
		x=0.5,
		y=0.5,
		xref='paper',
		yref='paper',
		showarrow=False,
		font=dict(size=14, color='#475569'),
	)
	fig.update_layout(
		title=title,
		xaxis=dict(visible=False),
		yaxis=dict(visible=False),
		margin=dict(l=20, r=20, t=50, b=20),
		plot_bgcolor='#f8fafc',
		paper_bgcolor='white',
	)
	return fig


def _safe_literal_assignments(code):
	assignments = {}
	if not code:
		return assignments

	try:
		parsed = ast.parse(code)
	except SyntaxError:
		return assignments

	for node in parsed.body:
		if not isinstance(node, ast.Assign) or len(node.targets) != 1:
			continue
		target = node.targets[0]
		if not isinstance(target, ast.Name):
			continue
		try:
			assignments[target.id] = ast.literal_eval(node.value)
		except Exception:
			continue
	return assignments


def _normalise_dataset(value):
	if isinstance(value, str) and value in {'moons', 'circles', 'linear'}:
		return value
	return DEFAULT_LEVEL3_META['dataset']


def _normalise_activation(value):
	if isinstance(value, str) and value in {'relu', 'tanh', 'sigmoid'}:
		return value
	return DEFAULT_LEVEL3_META['activation']


def _normalise_hidden_layers(value):
	if isinstance(value, int):
		return [max(2, min(16, int(value)))]
	if isinstance(value, (list, tuple)):
		cleaned = []
		for item in value:
			if isinstance(item, (int, float)):
				cleaned.append(max(2, min(16, int(item))))
		if cleaned:
			return cleaned[:4]
	return list(DEFAULT_LEVEL3_META['hidden_layer_sizes'])


def _normalise_positive_int(value, default_value, min_value, max_value):
	if isinstance(value, (int, float)):
		return max(min_value, min(max_value, int(value)))
	return default_value


def _normalise_positive_float(value, default_value, min_value, max_value):
	if isinstance(value, (int, float)):
		return max(min_value, min(max_value, float(value)))
	return default_value


def level3_extract_meta_from_code(cell_1_code, cell_2_code, cell_3_code, cell_4_code, cell_5_code, cell_6_code):
	cell_1_assignments = _safe_literal_assignments(cell_1_code)
	cell_2_assignments = _safe_literal_assignments(cell_2_code)
	cell_3_assignments = _safe_literal_assignments(cell_3_code)
	cell_4_assignments = _safe_literal_assignments(cell_4_code)
	cell_5_assignments = _safe_literal_assignments(cell_5_code)
	cell_6_assignments = _safe_literal_assignments(cell_6_code)

	dataset = _normalise_dataset(cell_1_assignments.get('dataset_name'))
	input_dim = _normalise_positive_int(
		cell_2_assignments.get('input_dim', DEFAULT_LEVEL3_META['input_dim']),
		DEFAULT_LEVEL3_META['input_dim'],
		2,
		8,
	)
	activation = _normalise_activation(cell_3_assignments.get('activation'))
	hidden_layer_sizes = _normalise_hidden_layers(cell_4_assignments.get('hidden_layers'))
	output_dim = _normalise_positive_int(
		cell_5_assignments.get('output_dim', DEFAULT_LEVEL3_META['output_dim']),
		DEFAULT_LEVEL3_META['output_dim'],
		1,
		1,
	)
	epochs = _normalise_positive_int(
		cell_6_assignments.get('epochs', DEFAULT_LEVEL3_META['epochs']),
		DEFAULT_LEVEL3_META['epochs'],
		1,
		400,
	)
	learning_rate = _normalise_positive_float(
		cell_6_assignments.get('learning_rate', DEFAULT_LEVEL3_META['learning_rate']),
		DEFAULT_LEVEL3_META['learning_rate'],
		0.001,
		1.0,
	)

	return level3_build_meta(dataset, hidden_layer_sizes, activation, epochs, learning_rate, input_dim, output_dim)


def level3_build_meta(dataset, hidden_layer_sizes, activation, epochs, learning_rate, input_dim=2, output_dim=1):
	hidden_layers = _normalise_hidden_layers(hidden_layer_sizes)
	return {
		'dataset': _normalise_dataset(dataset),
		'input_dim': _normalise_positive_int(input_dim, 2, 2, 8),
		'hidden_layer_sizes': hidden_layers,
		'activation': _normalise_activation(activation),
		'epochs': _normalise_positive_int(epochs, DEFAULT_LEVEL3_META['epochs'], 1, 400),
		'learning_rate': _normalise_positive_float(learning_rate, DEFAULT_LEVEL3_META['learning_rate'], 0.001, 1.0),
		'output_dim': 1,
		'depth': len(hidden_layers),
		'layer_sizes': [input_dim] + hidden_layers + [output_dim],
	}


def level3_build_dataset(meta):
	return build_level2_dataset(meta['dataset'], input_dim=meta['input_dim'])


def level3_initialise_store(meta):
	return {
		'meta': meta,
		'model': None,
		'cell_runs': {
			'load_dataset': False,
			'input_layer': False,
			'activation': False,
			'hidden_layers': False,
			'output_layer': False,
			'training_config': False,
		},
		'training_logs': [],
	}


def level3_model_matches(store, meta):
	model = store.get('model')
	if model is None:
		return False

	model_meta = model.get('meta', {})
	return (
		model_meta.get('hidden_layer_sizes') == meta['hidden_layer_sizes']
		and model_meta.get('activation') == meta['activation']
		and model_meta.get('dataset') == meta['dataset']
		and model_meta.get('input_dim') == meta['input_dim']
	)


def level3_initialise_model(store, meta):
	dataset_bundle = level3_build_dataset(meta)
	model = init_level2_mlp(
		input_dim=meta['input_dim'],
		hidden_layers=meta['hidden_layer_sizes'],
		output_dim=meta['output_dim'],
	)
	model['meta'] = {
		'hidden_layer_sizes': meta['hidden_layer_sizes'],
		'activation': meta['activation'],
		'dataset': meta['dataset'],
		'input_dim': meta['input_dim'],
		'output_dim': meta['output_dim'],
		'learning_rate': meta['learning_rate'],
		'layer_sizes': meta['layer_sizes'],
	}
	model = level2_set_baseline_history(dataset_bundle, model, meta['activation'], l2=1e-4)
	store['model'] = model
	store['training_logs'] = []
	return store


def level3_dataset_preview_figure(dataset_bundle):
	fig = go.Figure()
	split_specs = [
		('Train class 0', dataset_bundle['X_train_raw'][dataset_bundle['y_train'] == 0], '#0f766e', 'circle'),
		('Train class 1', dataset_bundle['X_train_raw'][dataset_bundle['y_train'] == 1], '#b91c1c', 'circle'),
		('Test class 0', dataset_bundle['X_test_raw'][dataset_bundle['y_test'] == 0], '#14b8a6', 'diamond-open'),
		('Test class 1', dataset_bundle['X_test_raw'][dataset_bundle['y_test'] == 1], '#f97316', 'diamond-open'),
	]

	for label, points, colour, symbol in split_specs:
		if len(points) == 0:
			continue
		fig.add_trace(go.Scatter(
			x=points[:, 0],
			y=points[:, 1],
			mode='markers',
			name=label,
			marker=dict(size=8, color=colour, symbol=symbol, line=dict(width=1, color='white')),
		))

	fig.update_layout(
		title=f"Dataset preview: {dataset_bundle['dataset_name'].title()} split into train/test batches",
		xaxis_title='x1',
		yaxis_title='x2',
		margin=dict(l=30, r=10, t=50, b=30),
		legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
	)
	return fig


def level3_activation_heatmap_figure(model, X_reference, activation):
	_, cache = level2_forward_pass(X_reference, model, activation)
	hidden_activations = cache['activations'][1:-1]
	if not hidden_activations:
		return make_level3_placeholder_figure('Per-layer activations', 'Define a model with at least one hidden layer.')

	fig = make_subplots(
		rows=len(hidden_activations),
		cols=1,
		shared_xaxes=True,
		vertical_spacing=0.08,
		subplot_titles=[f'Hidden layer {index + 1}' for index in range(len(hidden_activations))],
	)

	for index, activation_matrix in enumerate(hidden_activations, start=1):
		z_values = activation_matrix[:min(12, activation_matrix.shape[0]), :min(80, activation_matrix.shape[1])]
		fig.add_trace(
			go.Heatmap(
				z=z_values,
				colorscale='RdBu',
				zmid=0,
				showscale=(index == 1),
				colorbar=dict(title='Activation') if index == 1 else None,
			),
			row=index,
			col=1,
		)
		fig.update_yaxes(title_text='Neuron', row=index, col=1)

	fig.update_xaxes(title_text='Sample index', row=len(hidden_activations), col=1)
	fig.update_layout(
		title='Per-layer activations across a held-out batch',
		height=max(280, 220 * len(hidden_activations)),
		margin=dict(l=40, r=10, t=60, b=30),
	)
	return fig


def level3_hidden_space_figure(model, X_reference, y_reference, activation):
	_, cache = level2_forward_pass(X_reference, model, activation)
	hidden_activations = cache['activations'][1:-1]
	if not hidden_activations:
		return make_level3_placeholder_figure('Hidden-space projection', 'Run Cell 5 to inspect hidden layers.')

	last_hidden = hidden_activations[-1].T
	x_axis = last_hidden[:, 0]
	y_axis = last_hidden[:, 1] if last_hidden.shape[1] > 1 else np.zeros(last_hidden.shape[0])

	fig = go.Figure()
	for class_value, colour in [(0, '#0f766e'), (1, '#b91c1c')]:
		mask = y_reference == class_value
		fig.add_trace(go.Scatter(
			x=x_axis[mask],
			y=y_axis[mask],
			mode='markers',
			name=f'Class {class_value}',
			marker=dict(size=9, color=colour, line=dict(width=1, color='white')),
		))

	fig.update_layout(
		title='Last hidden layer projection (neurons 1 and 2)',
		xaxis_title='Hidden dimension 1',
		yaxis_title='Hidden dimension 2',
		margin=dict(l=30, r=10, t=50, b=30),
	)
	return fig


def level3_confusion_matrix_figure(confusion_values):
	labels = ['Class 0', 'Class 1']
	fig = go.Figure(data=go.Heatmap(
		z=confusion_values,
		x=labels,
		y=labels,
		text=confusion_values,
		texttemplate='%{text}',
		textfont={'size': 14},
		colorscale='Blues',
		hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>',
	))
	fig.update_layout(
		title='Confusion matrix on the evaluation split',
		xaxis_title='Predicted label',
		yaxis_title='True label',
		margin=dict(l=80, r=20, t=55, b=40),
	)
	return fig


def level3_misclassified_figure(dataset_bundle, pred_labels):
	X_test = dataset_bundle['X_test_raw']
	y_test = dataset_bundle['y_test']
	misclassified = pred_labels != y_test
	fig = go.Figure()
	for class_value, colour in [(0, '#94a3b8'), (1, '#475569')]:
		mask = y_test == class_value
		fig.add_trace(go.Scatter(
			x=X_test[mask, 0],
			y=X_test[mask, 1],
			mode='markers',
			name=f'Test class {class_value}',
			marker=dict(size=7, color=colour),
		))

	if np.any(misclassified):
		fig.add_trace(go.Scatter(
			x=X_test[misclassified, 0],
			y=X_test[misclassified, 1],
			mode='markers',
			name='Misclassified',
			marker=dict(size=11, color='#ef4444', symbol='x', line=dict(width=2)),
		))

	fig.update_layout(
		title='Misclassified evaluation samples',
		xaxis_title='x1',
		yaxis_title='x2',
		margin=dict(l=30, r=10, t=50, b=30),
	)
	return fig


def level3_forward_summary_children(forward_summary):
	if not forward_summary:
		return html.Div('Run Cell 3 to inspect tensor shapes and example predictions.', style={'color': '#64748b'})

	example_rows = [
		html.Tr([
			html.Td(f"({example['x1']:.2f}, {example['x2']:.2f})"),
			html.Td(str(example['target'])),
			html.Td(f"{example['probability']:.3f}"),
		])
		for example in forward_summary['examples']
	]

	return html.Div([
		html.P(f"Batch shape: {tuple(forward_summary['batch_shape'])}"),
		html.P(f"Output shape: {tuple(forward_summary['output_shape'])}"),
		html.Ul([
			html.Li(f"Hidden layer {index + 1}: {tuple(shape)}")
			for index, shape in enumerate(forward_summary['hidden_shapes'])
		], style={'fontFamily': 'monospace', 'fontSize': '12px'}),
		html.Table(
			[
				html.Thead(
					html.Tr([
						html.Th('Input sample'),
						html.Th('Target'),
						html.Th('Predicted p(class=1)'),
					])
				),
				html.Tbody(example_rows),
			],
			style={'width': '100%', 'fontSize': '12px'},
		),
	])


def level3_training_log_children(training_logs):
	if not training_logs:
		return html.Div('Run Cell 4 to train the classifier and capture a notebook-style training log.', style={'color': '#64748b'})

	return html.Ul([
		html.Li(
			f"Run {entry['run_number']}: epochs={entry['epochs']}, lr={entry['learning_rate']:.3f}, "
			f"train loss={entry['train_loss']:.4f}, train acc={entry['train_accuracy'] * 100:.1f}%, "
			f"test loss={entry['test_loss']:.4f}, test acc={entry['test_accuracy'] * 100:.1f}%"
		)
		for entry in reversed(training_logs)
	], style={'paddingLeft': '18px', 'fontSize': '12px'})


def level3_metrics_summary_children(evaluation):
	if not evaluation:
		return html.Div('Run Cell 6 to compute confusion, metrics, and misclassified samples.', style={'color': '#64748b'})

	metrics = evaluation['metrics']
	precision = evaluation['precision']
	recall = evaluation['recall']
	f1 = evaluation['f1']
	support = evaluation['support']
	rows = []
	for index in range(len(precision)):
		rows.append(html.Tr([
			html.Td(f'Class {index}'),
			html.Td(f'{precision[index]:.2f}'),
			html.Td(f'{recall[index]:.2f}'),
			html.Td(f'{f1[index]:.2f}'),
			html.Td(str(support[index])),
		]))

	return html.Div([
		html.P(f"Train accuracy: {metrics['train_accuracy'] * 100:.1f}%"),
		html.P(f"Test accuracy: {metrics['test_accuracy'] * 100:.1f}%"),
		html.P(f"Train loss: {metrics['train_loss']:.4f}"),
		html.P(f"Test loss: {metrics['test_loss']:.4f}"),
		html.P(f"Misclassified points: {evaluation['misclassified_count']} / {evaluation['sample_count']}"),
		html.Table([
			html.Thead(html.Tr([
				html.Th('Class'), html.Th('Precision'), html.Th('Recall'), html.Th('F1'), html.Th('Support'),
			])),
			html.Tbody(rows),
		], style={'width': '100%', 'fontSize': '12px'}),
	])


def level3_dataset_summary_children(dataset_bundle, meta):
	return html.Div([
		html.P(f"Dataset: {meta['dataset'].title()}"),
		html.P(
			f"Train split: {dataset_bundle['X_train'].shape[0]} samples | "
			f"Test split: {dataset_bundle['X_test'].shape[0]} samples"
		),
		html.P(
			f"Class balance (train): class 0 = {int(np.sum(dataset_bundle['y_train'] == 0))}, "
			f"class 1 = {int(np.sum(dataset_bundle['y_train'] == 1))}"
		),
		html.P(
			f"Feature layer width: {meta['input_dim']} | "
			f"Hidden stack: {meta['hidden_layer_sizes']} | Output dim: {meta['output_dim']}"
		),
		html.P(
			f"Cached training plan: {meta['epochs']} epochs at learning rate {meta['learning_rate']:.3f}"
		),
	], style={'fontSize': '12px'})


def level3_setup_complete(store):
	return bool(store) and all(store.get('cell_runs', {}).values())


def level3_notebook_status_children(store):
	if store is None or not store['cell_runs']['load_dataset']:
		return html.Div('Start with Cell 1 to load the dataset and populate the output box beside it.')
	if not store['cell_runs']['input_layer']:
		return html.Div('Dataset committed. Run Cell 2 to define the input layer width.')
	if not store['cell_runs']['activation']:
		return html.Div('Input layer committed. Run Cell 3 to choose the activation function.')
	if not store['cell_runs']['hidden_layers']:
		return html.Div('Activation committed. Run Cell 4 to define the hidden-layer stack.')
	if not store['cell_runs']['output_layer']:
		return html.Div('Hidden stack committed. Run Cell 5 to define the output layer.')
	if not store['cell_runs']['training_config']:
		return html.Div('Structure committed. Run Cell 6 to lock in epochs and learning rate for cached training playback.')
	if store.get('model') is None or store['model'].get('epoch', 0) == 0:
		return html.Div('All setup cells are complete. Use the training controls below to animate the cached epoch stages.')
	return html.Div('Cached training is available. Use Start, Pause, Auto/Semi-Auto, or stage stepping to inspect the current epoch timeline.')


def level3_model_status_children(store):
	if store is None or store.get('model') is None:
		status = 'Waiting for model definition'
		color = '#64748b'
		detail = 'Run the structure cells to build the code-defined network.'
	else:
		epoch = int(store['model'].get('epoch', 0))
		if epoch > 0:
			status = 'Training cached'
			color = '#0f766e'
			detail = f'Epochs completed: {epoch}'
		else:
			status = 'Model ready'
			color = '#1d4ed8'
			detail = 'Architecture initialised and ready for cached training playback.'

	return html.Div([
		html.Div('Model Status', style={'fontSize': '10px', 'textTransform': 'uppercase', 'letterSpacing': '0.08em', 'color': '#64748b', 'marginBottom': '4px'}),
		html.Div(status, style={'fontSize': '19px', 'fontWeight': '700', 'color': color, 'marginBottom': '4px'}),
		html.Div(detail, style={'fontSize': '12px', 'color': '#475569'}),
	])


def level3_execution_live_children(store):
	if store is None:
		completed = 0
	else:
		completed = sum(1 for value in store.get('cell_runs', {}).values() if value)
	ready_text = 'Training controls unlock after 6 / 6 setup cells are committed.'
	if level3_setup_complete(store):
		ready_text = 'Setup complete. The training and animation controls now operate on the cached epoch stages.'

	return html.Div([
		html.Div('Notebook Progress', style={'fontSize': '10px', 'textTransform': 'uppercase', 'letterSpacing': '0.08em', 'color': '#64748b', 'marginBottom': '4px'}),
		html.Div(f'{completed} / 6 cells run', style={'fontSize': '19px', 'fontWeight': '700', 'color': '#0f172a', 'marginBottom': '4px'}),
		html.Div(ready_text, style={'fontSize': '12px', 'color': '#475569'}),
	])


def build_level3_execution_environment(cell_number, meta):
	dataset_bundle = level3_build_dataset(meta)
	model = init_level2_mlp(
		input_dim=meta['input_dim'],
		hidden_layers=meta['hidden_layer_sizes'],
		output_dim=meta['output_dim'],
	)
	model = level2_set_baseline_history(dataset_bundle, model, meta['activation'], l2=1e-4)

	return {
		'__builtins__': {
			'print': print,
			'int': int,
			'float': float,
			'str': str,
			'list': list,
			'dict': dict,
			'tuple': tuple,
			'range': range,
			'len': len,
			'sum': sum,
			'max': max,
			'min': min,
			'__import__': __import__,
		},
		'np': np,
		'confusion_matrix': confusion_matrix,
		'build_level2_dataset': build_level2_dataset,
		'init_level2_mlp': init_level2_mlp,
		'level2_forward_pass': level2_forward_pass,
		'level2_evaluate_metrics': level2_evaluate_metrics,
		'level2_set_baseline_history': level2_set_baseline_history,
		'train_level2_model': train_level2_model,
		'dataset_name': meta['dataset'],
		'input_dim': meta['input_dim'],
		'hidden_layers': meta['hidden_layer_sizes'],
		'activation': meta['activation'],
		'epochs': meta['epochs'],
		'learning_rate': meta['learning_rate'],
		'training_config': {'epochs': meta['epochs'], 'learning_rate': meta['learning_rate']},
		'dataset_bundle': dataset_bundle,
		'X': dataset_bundle['X_raw'],
		'y': dataset_bundle['y'],
		'X_train': dataset_bundle['X_train'],
		'X_test': dataset_bundle['X_test'],
		'X_train_raw': dataset_bundle['X_train_raw'],
		'X_test_raw': dataset_bundle['X_test_raw'],
		'y_train': dataset_bundle['y_train'],
		'y_test': dataset_bundle['y_test'],
		'model': model,
	}