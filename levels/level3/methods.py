import numpy as np
import plotly.graph_objects as go
from dash import html
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from levels.level2 import (
	init_level2_mlp,
	level2_evaluate_metrics,
	level2_forward_pass,
	level2_set_baseline_history,
	load_toy_dataset,
	train_level2_model,
)


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


def level3_build_meta(dataset, depth, width, activation, epochs):
	hidden_layer_sizes = [width] * depth
	return {
		'dataset': dataset,
		'depth': depth,
		'width': width,
		'hidden_layer_sizes': hidden_layer_sizes,
		'activation': activation,
		'epochs': epochs,
		'layer_sizes': [2] + hidden_layer_sizes + [1],
	}


def level3_serialize_split(X_train, X_test, y_train, y_test, X_full, y_full):
	return {
		'X_train': X_train.tolist(),
		'X_test': X_test.tolist(),
		'y_train': y_train.tolist(),
		'y_test': y_test.tolist(),
		'X_full': X_full.tolist(),
		'y_full': y_full.tolist(),
	}


def level3_deserialize_split(data):
	return (
		np.array(data['X_train'], dtype=np.float64),
		np.array(data['X_test'], dtype=np.float64),
		np.array(data['y_train'], dtype=np.int32),
		np.array(data['y_test'], dtype=np.int32),
		np.array(data['X_full'], dtype=np.float64),
		np.array(data['y_full'], dtype=np.int32),
	)


def level3_initialize_store(meta):
	X_full, y_full = load_toy_dataset(meta['dataset'])
	X_train, X_test, y_train, y_test = train_test_split(
		X_full,
		y_full,
		test_size=0.25,
		random_state=42,
		stratify=y_full,
	)
	return {
		'meta': meta,
		'data': level3_serialize_split(X_train, X_test, y_train, y_test, X_full, y_full),
		'model': None,
		'cell_runs': {
			'load_dataset': False,
			'define_model': False,
			'forward_pass': False,
			'train_model': False,
			'inspect': False,
			'evaluate': False,
		},
		'forward_summary': None,
		'training_logs': [],
		'evaluation': None,
		'inspect_ran': False,
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
	)


def level3_initialize_model(store, meta):
	X_train, _, y_train, _, _, _ = level3_deserialize_split(store['data'])
	model = init_level2_mlp(
		input_dim=2,
		hidden_layers=meta['hidden_layer_sizes'],
		output_dim=1,
	)
	model['meta'] = {
		'hidden_layers': meta['depth'],
		'neurons_per_layer': meta['width'],
		'hidden_layer_sizes': meta['hidden_layer_sizes'],
		'activation': meta['activation'],
		'dataset': meta['dataset'],
		'layer_sizes': meta['layer_sizes'],
	}
	model = level2_set_baseline_history(X_train, y_train, model, meta['activation'], l2=1e-4)
	store['model'] = model
	store['forward_summary'] = None
	store['evaluation'] = None
	store['inspect_ran'] = False
	store['cell_runs']['forward_pass'] = False
	store['cell_runs']['train_model'] = False
	store['cell_runs']['inspect'] = False
	store['cell_runs']['evaluate'] = False
	return store


def level3_dataset_preview_figure(X_train, X_test, y_train, y_test, dataset):
	fig = go.Figure()
	split_specs = [
		('Train class 0', X_train[y_train == 0], '#0f766e', 'circle'),
		('Train class 1', X_train[y_train == 1], '#b91c1c', 'circle'),
		('Test class 0', X_test[y_test == 0], '#14b8a6', 'diamond-open'),
		('Test class 1', X_test[y_test == 1], '#f97316', 'diamond-open'),
	]

	for label, points, color, symbol in split_specs:
		if len(points) == 0:
			continue
		fig.add_trace(go.Scatter(
			x=points[:, 0],
			y=points[:, 1],
			mode='markers',
			name=label,
			marker=dict(size=8, color=color, symbol=symbol, line=dict(width=1, color='white')),
		))

	fig.update_layout(
		title=f'Dataset preview: {dataset.title()} split into train/test batches',
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
		return make_level3_placeholder_figure('Hidden-space projection', 'Run Cell 2 to define hidden layers.')

	last_hidden = hidden_activations[-1].T
	x_axis = last_hidden[:, 0]
	y_axis = last_hidden[:, 1] if last_hidden.shape[1] > 1 else np.zeros(last_hidden.shape[0])

	fig = go.Figure()
	for class_value, color in [(0, '#0f766e'), (1, '#b91c1c')]:
		mask = y_reference == class_value
		fig.add_trace(go.Scatter(
			x=x_axis[mask],
			y=y_axis[mask],
			mode='markers',
			name=f'Class {class_value}',
			marker=dict(size=9, color=color, line=dict(width=1, color='white')),
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


def level3_misclassified_figure(X_test, y_test, pred_labels):
	misclassified = pred_labels != y_test
	fig = go.Figure()
	for class_value, color in [(0, '#94a3b8'), (1, '#475569')]:
		mask = y_test == class_value
		fig.add_trace(go.Scatter(
			x=X_test[mask, 0],
			y=X_test[mask, 1],
			mode='markers',
			name=f'Test class {class_value}',
			marker=dict(size=7, color=color),
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
			f"Run {entry['run_number']}: epochs={entry['epochs']}, train loss={entry['train_loss']:.4f}, "
			f"train acc={entry['train_accuracy'] * 100:.1f}%, test loss={entry['test_loss']:.4f}, "
			f"test acc={entry['test_accuracy'] * 100:.1f}%"
		)
		for entry in reversed(training_logs)
	], style={'paddingLeft': '18px', 'fontSize': '12px'})


def level3_metrics_summary_children(evaluation):
	if not evaluation:
		return html.Div('Run Cell 6 to compute confusion, metrics, and misclassified samples.', style={'color': '#64748b'})

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
		html.P(f"Accuracy: {evaluation['metrics']['accuracy'] * 100:.1f}%"),
		html.P(f"Loss: {evaluation['metrics']['loss']:.4f}"),
		html.P(f"Misclassified points: {evaluation['misclassified_count']} / {evaluation['sample_count']}"),
		html.Table([
			html.Thead(html.Tr([
				html.Th('Class'), html.Th('Precision'), html.Th('Recall'), html.Th('F1'), html.Th('Support'),
			])),
			html.Tbody(rows),
		], style={'width': '100%', 'fontSize': '12px'}),
	])


def level3_dataset_summary_children(X_train, X_test, y_train, y_test, meta):
	return html.Div([
		html.P(f"Dataset: {meta['dataset'].title()}"),
		html.P(f'Train split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples'),
		html.P(f'Class balance (train): class 0 = {int(np.sum(y_train == 0))}, class 1 = {int(np.sum(y_train == 1))}'),
		html.P(f"Feature space: 2-D input, {meta['depth']} hidden layer(s), {meta['width']} neuron(s) per hidden layer"),
	], style={'fontSize': '12px'})


def level3_notebook_status_children(store):
	if store is None or not store['cell_runs']['load_dataset']:
		return html.Div('Start with Cell 1 to load a dataset and preview the classification split.')
	if not store['cell_runs']['define_model']:
		return html.Div('Dataset loaded. Next run Cell 2 to define the network before inspecting any activations.')
	if not store['cell_runs']['forward_pass']:
		return html.Div('Model defined. Cell 3 is the next useful step if you want to inspect tensor shapes before training.')
	if not store['cell_runs']['train_model']:
		return html.Div('Forward pass captured. Run Cell 4 to optimise the model and update the boundary.')
	if not store['cell_runs']['inspect']:
		return html.Div('Training complete. Run Cell 5 to examine hidden representations and per-layer activations.')
	if not store['cell_runs']['evaluate']:
		return html.Div('Inspection complete. Run Cell 6 to compute evaluation metrics and misclassified points.')
	return html.Div('All six notebook steps have been executed. Change a configuration and rerun the relevant cell to compare behaviours.')


def build_level3_execution_environment(cell_number, dataset, depth, width, activation, epochs):
	hidden_layers = [width] * depth
	X, y = load_toy_dataset(dataset)
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.25,
		random_state=42,
		stratify=y,
	)

	model = init_level2_mlp(input_dim=2, hidden_layers=hidden_layers, output_dim=1)
	model = level2_set_baseline_history(X_train, y_train, model, activation, l2=1e-4)
	if cell_number >= 4:
		model = train_level2_model(
			X_train,
			y_train,
			model,
			activation=activation,
			epochs=epochs,
			lr=0.08,
			l2=1e-4,
		)

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
		'load_toy_dataset': load_toy_dataset,
		'init_level2_mlp': init_level2_mlp,
		'level2_forward_pass': level2_forward_pass,
		'level2_evaluate_metrics': level2_evaluate_metrics,
		'train_level2_model': train_level2_model,
		'dataset_name': dataset,
		'hidden_layers': hidden_layers,
		'activation': activation,
		'epochs': epochs,
		'X': X,
		'y': y,
		'X_train': X_train,
		'X_test': X_test,
		'y_train': y_train,
		'y_test': y_test,
		'model': model,
	}
