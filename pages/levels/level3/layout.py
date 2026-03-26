from dash import dcc, html

from pySecProgramming.code_execution import CodeExecutionBox


PAGE_STYLE = {
	'padding': '8px 0 24px 0',
}

CARD_STYLE = {
	'backgroundColor': '#ffffff',
	'border': '1px solid #d7e3f4',
	'borderRadius': '18px',
	'padding': '16px',
	'boxShadow': '0 14px 34px rgba(15, 23, 42, 0.08)',
	'boxSizing': 'border-box',
}


LEVEL3_CELL_SNIPPETS = {
	1: (
		'# Cell 1: Load dataset\n'
		'dataset_name = "moons"\n'
		'dataset_bundle = build_level2_dataset(dataset_name, input_dim=2)\n'
		'print(f"Dataset: {dataset_name}")\n'
		'print("Train shape:", dataset_bundle["X_train"].shape)\n'
		'print("Test shape:", dataset_bundle["X_test"].shape)\n'
		'print("Class balance:", {0: int((dataset_bundle["y_train"] == 0).sum()), 1: int((dataset_bundle["y_train"] == 1).sum())})'
	),
	2: (
		'# Cell 2: Define model\n'
		'input_dim = 2\n'
		'hidden_layers = [6, 6]\n'
		'activation = "tanh"\n'
		'output_dim = 1\n'
		'model = init_level2_mlp(input_dim=input_dim, hidden_layers=hidden_layers, output_dim=output_dim)\n'
		'model = level2_set_baseline_history(dataset_bundle, model, activation, l2=1e-4)\n'
		'print("Activation:", activation)\n'
		'print("Hidden layers:", hidden_layers)\n'
		'print("Weight shapes:", [np.array(weight).shape for weight in model["weights"]])'
	),
	3: (
		'# Cell 3: Forward pass\n'
		'batch_X = dataset_bundle["X_train"][:24]\n'
		'batch_y = dataset_bundle["y_train"][:24]\n'
		'probs, cache = level2_forward_pass(batch_X, model, activation)\n'
		'print("Batch shape:", batch_X.shape)\n'
		'print("Output shape:", probs.shape)\n'
		'print("Hidden shapes:", [layer.shape for layer in cache["activations"][1:-1]])\n'
		'print("First 5 probabilities:", probs.flatten()[:5])\n'
		'print("First 5 targets:", batch_y[:5])'
	),
	4: (
		'# Cell 4: Train model\n'
		'epochs = 120\n'
		'learning_rate = 0.08\n'
		'model = train_level2_model(dataset_bundle, model, activation=activation, epochs=epochs, lr=learning_rate, l2=1e-4)\n'
		'metrics = level2_evaluate_metrics(dataset_bundle, model, activation, l2=1e-4)\n'
		'print("Epoch:", model["epoch"])\n'
		'print("Train accuracy:", round(metrics["train_accuracy"], 4))\n'
		'print("Test accuracy:", round(metrics["test_accuracy"], 4))\n'
		'print("Train loss:", round(metrics["train_loss"], 4))\n'
		'print("Test loss:", round(metrics["test_loss"], 4))'
	),
	5: (
		'# Cell 5: Inspect internals\n'
		'_, cache = level2_forward_pass(dataset_bundle["X_test"][:24], model, activation)\n'
		'print("Hidden layers:", len(cache["activations"]) - 2)\n'
		'print("Hidden activation shapes:", [layer.shape for layer in cache["activations"][1:-1]])'
	),
	6: (
		'# Cell 6: Evaluate model\n'
		'probs, _ = level2_forward_pass(dataset_bundle["X_test"], model, activation)\n'
		'pred_labels = (probs.flatten() >= 0.5).astype(int)\n'
		'print("Accuracy:", float((pred_labels == dataset_bundle["y_test"]).mean()))\n'
		'print("Confusion matrix:\n", confusion_matrix(dataset_bundle["y_test"], pred_labels))'
	),
}


LEVEL3_CELL_EDITORS = {
	cell_number: CodeExecutionBox(
		f'level3-cell-{cell_number}',
		ids={
			'input': f'level3-cell-{cell_number}-code',
			'run': button_id,
			'output': f'level3-cell-{cell_number}-console',
			'error': f'level3-cell-{cell_number}-error',
			'validation': f'level3-cell-{cell_number}-validation',
			'highlighted': f'level3-cell-{cell_number}-highlighted',
			'plot': f'level3-cell-{cell_number}-plot',
		},
	)
	for cell_number, button_id in {
		1: 'level3-load-data-btn',
		2: 'level3-define-model-btn',
		3: 'level3-forward-btn',
		4: 'level3-train-btn',
		5: 'level3-inspect-btn',
		6: 'level3-evaluate-btn',
	}.items()
}


def _level3_code_cell(cell_number, title, description, button_text):
	return LEVEL3_CELL_EDITORS[cell_number].render(
		default_code=LEVEL3_CELL_SNIPPETS[cell_number],
		title=title,
		description=description,
		controls=[],
		run_label=button_text,
		show_export=False,
		include_plot=False,
		code_height='122px',
		output_placeholder='Run the cell to see console output.',
		wrapper_style={'marginBottom': '12px'},
	)


def _level3_output_card(title, children):
	return html.Div(
		[
			html.H3(title, style={'marginTop': '0', 'marginBottom': '12px', 'fontSize': '16px'}),
			children,
		],
		style={**CARD_STYLE, 'height': '100%'},
	)


def level3_layout():
	return html.Div([
		html.H2(
			'Level 3 - Notebook Workflow for Building a Classifier',
			style={'textAlign': 'center', 'marginBottom': '8px', 'fontSize': '28px'},
		),
		html.P(
			'Level 3 now follows the same overall structure as Level 2, but model architecture and training settings are defined by editable code cells instead of sliders.',
			style={
				'textAlign': 'center',
				'margin': '0 auto 18px auto',
				'maxWidth': '1040px',
				'color': '#475569',
				'lineHeight': '1.5',
				'fontSize': '14px',
			},
		),
		html.Div([
			html.Div(id='level3-notebook-status', style={'flex': '1 1 320px', 'padding': '12px 14px', 'borderRadius': '14px', 'backgroundColor': '#f8fafc', 'border': '1px solid #dbeafe', 'minWidth': '260px'}),
			html.Div(id='level3-model-status', style={'flex': '0.9 1 220px', 'padding': '12px 14px', 'borderRadius': '14px', 'backgroundColor': '#f8fafc', 'border': '1px solid #dbeafe', 'minWidth': '220px'}),
			html.Div(id='level3-execution-live', style={'flex': '0.9 1 220px', 'padding': '12px 14px', 'borderRadius': '14px', 'backgroundColor': '#f8fafc', 'border': '1px solid #dbeafe', 'minWidth': '220px'}),
		], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'alignItems': 'stretch', 'marginBottom': '16px'}),
		html.Div([
			html.Div([
				html.Div([
					html.H3('Dataset Handling', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
					html.P('Cell 1 replaces a dataset dropdown. Edit the dataset name in code, run it, and the rest of the workflow uses that configuration.', style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45', 'marginBottom': '10px'}),
					_level3_code_cell(1, '1. Load Dataset', 'Choose the toy dataset directly in code.', 'Run Cell 1'),
					html.Div(id='level3-dataset-summary'),
				], style=CARD_STYLE),
				html.Div([
					html.H3('Notebook Controls', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
					html.P('Cells 2 and 4 replace Level 2 sliders. Define hidden layers, activation, epochs, and learning rate in code, then run the cell to update the visuals.', style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45', 'marginBottom': '10px'}),
					_level3_code_cell(2, '2. Define Model', 'Set input_dim, hidden_layers, activation, and output_dim in code.', 'Run Cell 2'),
					_level3_code_cell(3, '3. Forward Pass', 'Inspect tensor shapes and example outputs before training.', 'Run Cell 3'),
					_level3_code_cell(4, '4. Train Model', 'Set epochs and learning_rate in code, then train the model.', 'Run Cell 4'),
					_level3_code_cell(5, '5. Inspect Internals', 'Inspect hidden activations after training.', 'Run Cell 5'),
					_level3_code_cell(6, '6. Evaluate Model', 'Evaluate the trained classifier and inspect confusion.', 'Run Cell 6'),
				], style={**CARD_STYLE, 'marginTop': '14px'}),
			], style={'flex': '1 1 320px', 'minWidth': '320px', 'maxWidth': '420px'}),
			html.Div([
				html.Div([
					html.H3('FNN Architecture', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
					html.P('The architecture view is central just like Level 2. Cell 2 determines the hidden stack and activation, and Cell 4 updates the trained parameters.', style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45', 'marginBottom': '14px'}),
					dcc.Graph(id='level3-network-diagram-graph', style={'height': '52vh'}),
					html.Div(id='level3-arch-summary', style={'marginTop': '10px'}),
				], style=CARD_STYLE),
			], style={'flex': '1.55 1 620px', 'minWidth': '480px'}),
			html.Div([
				html.Div(id='level3-output-summary', style={**CARD_STYLE, 'padding': '10px 12px', 'marginBottom': '14px'}),
				html.Div([
					html.H3('Decision Boundary', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
					html.P('Inspect the learned boundary over the original two-dimensional input space.', style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45'}),
					dcc.Graph(id='level3-boundary-graph', style={'height': '44vh'}),
				], style=CARD_STYLE),
				html.Div([
					html.H3('Training Curves', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
					html.P('Cell 4 writes optimisation history, which is shown here using the same metric vocabulary as Level 2.', style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45'}),
					dcc.Graph(id='level3-loss-graph', style={'height': '28vh'}),
				], style={**CARD_STYLE, 'marginTop': '14px'}),
			], style={'flex': '1 1 320px', 'minWidth': '300px'}),
		], style={'display': 'flex', 'gap': '20px', 'alignItems': 'flex-start', 'flexWrap': 'wrap'}),
		html.H3('Notebook Outputs', style={'margin': '18px 0 14px 0', 'fontSize': '20px'}),
		html.Div([
			html.Div([
				_level3_output_card('Cell 1 Output - Dataset Preview', dcc.Graph(id='level3-dataset-preview-graph', style={'height': '30vh'})),
			], style={'flex': '1 1 48%', 'minWidth': '320px'}),
			html.Div([
				_level3_output_card('Cell 3 Output - Forward Pass', html.Div(id='level3-forward-output')),
			], style={'flex': '1 1 48%', 'minWidth': '320px'}),
			html.Div([
				_level3_output_card('Cell 4 Output - Training Log', html.Div(id='level3-training-log')),
			], style={'flex': '1 1 48%', 'minWidth': '320px'}),
			html.Div([
				_level3_output_card('Cell 5 Output - Activation Heatmap', dcc.Graph(id='level3-activations-graph', style={'height': '34vh'})),
			], style={'flex': '1 1 48%', 'minWidth': '320px'}),
			html.Div([
				_level3_output_card('Cell 5 Output - Hidden Representations', dcc.Graph(id='level3-hidden-space-graph', style={'height': '34vh'})),
			], style={'flex': '1 1 48%', 'minWidth': '320px'}),
			html.Div([
				_level3_output_card(
					'Cell 6 Output - Evaluation',
					html.Div([
						dcc.Graph(id='level3-confusion-matrix-graph', style={'height': '26vh'}),
						dcc.Graph(id='level3-misclassified-graph', style={'height': '26vh'}),
						html.Div(id='level3-metrics-summary'),
					]),
				),
			], style={'flex': '1 1 48%', 'minWidth': '320px'}),
		], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '16px'}),
		dcc.Store(id='level3-params-store'),
	], style=PAGE_STYLE)