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

STATUS_CARD_STYLE = {
	'padding': '12px 14px',
	'borderRadius': '14px',
	'backgroundColor': '#f8fafc',
	'border': '1px solid #dbeafe',
	'minWidth': '220px',
}

CONTROL_BUTTON_STYLE = {
	'border': 'none',
	'padding': '10px 14px',
	'borderRadius': '10px',
	'fontWeight': '600',
	'fontSize': '13px',
	'cursor': 'pointer',
}


LEVEL3_CELL_SNIPPETS = {
	1: (
		'# Cell 1: Dataset handling\n'
		'dataset_name = "moons"\n'
		'dataset_bundle = build_level2_dataset(dataset_name, input_dim=2)\n'
		'print(f"Dataset: {dataset_name}")\n'
		'print("Train shape:", dataset_bundle["X_train"].shape)\n'
		'print("Test shape:", dataset_bundle["X_test"].shape)'
	),
	2: (
		'# Cell 2: Input layer structure\n'
		'input_dim = 2\n'
		'input_features = ["x1", "x2"]\n'
		'print("Input layer width:", input_dim)\n'
		'print("Feature labels:", input_features)'
	),
	3: (
		'# Cell 3: Activation function structure\n'
		'activation = "tanh"\n'
		'print("Hidden activation:", activation)\n'
		'print("Output head: sigmoid")'
	),
	4: (
		'# Cell 4: Hidden layer structure\n'
		'hidden_layers = [6, 6]\n'
		'print("Hidden layers:", hidden_layers)\n'
		'print("Network depth:", len(hidden_layers))'
	),
	5: (
		'# Cell 5: Output layer structure\n'
		'output_dim = 1\n'
		'output_name = "binary probability"\n'
		'print("Output layer width:", output_dim)\n'
		'print("Prediction meaning:", output_name)'
	),
	6: (
		'# Cell 6: Training configuration\n'
		'epochs = 120\n'
		'learning_rate = 0.08\n'
		'print("Cached training epochs:", epochs)\n'
		'print("Learning rate:", learning_rate)\n'
		'print("Training controls below will animate the cached epoch stages.")'
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
		2: 'level3-input-layer-btn',
		3: 'level3-activation-btn',
		4: 'level3-hidden-layers-btn',
		5: 'level3-output-layer-btn',
		6: 'level3-training-config-btn',
	}.items()
}


def _level3_code_cell(cell_number, title, description, button_text, *, code_height='116px'):
	return LEVEL3_CELL_EDITORS[cell_number].render(
		default_code=LEVEL3_CELL_SNIPPETS[cell_number],
		title=title,
		description=description,
		controls=[],
		run_label=button_text,
		show_export=False,
		include_plot=False,
		code_height=code_height,
		output_placeholder='Run the cell to see console output.',
		wrapper_style={'marginBottom': '12px'},
	)


def level3_layout():
	return html.Div([
		html.H2(
			'Level 3 - Code-Driven FNN Assembly',
			style={'textAlign': 'center', 'marginBottom': '8px', 'fontSize': '28px'},
		),
		html.P(
			'Level 3 now separates dataset setup, network structure, and cached training playback. Run the code cells in order, then use the controls to animate the saved forward, loss, backward, and update stages.',
			style={
				'textAlign': 'center',
				'margin': '0 auto 18px auto',
				'maxWidth': '1080px',
				'color': '#475569',
				'lineHeight': '1.5',
				'fontSize': '14px',
			},
		),
		html.Div([
			html.Div(id='level3-notebook-status', style={'flex': '1.2 1 300px', **STATUS_CARD_STYLE}),
			html.Div(id='level3-model-status', style={'flex': '0.9 1 220px', **STATUS_CARD_STYLE}),
			html.Div(id='level3-execution-live', style={'flex': '0.9 1 220px', **STATUS_CARD_STYLE}),
			html.Div(id='level3-training-stage-panel', style={'flex': '1 1 260px', **STATUS_CARD_STYLE}),
		], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'alignItems': 'stretch', 'marginBottom': '18px'}),
		html.Div([
			html.Div([
				html.H3('Dataset Handling', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
				html.P('Choose the toy dataset in code. This is the only setup cell that must run before the structure cells.', style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45', 'marginBottom': '10px'}),
				_level3_code_cell(1, '1. Dataset Cell', 'Load a dataset bundle that the structure and training stages will reuse.', 'Run Dataset Cell', code_height='112px'),
			], style={**CARD_STYLE, 'flex': '1 1 420px', 'minWidth': '340px'}),
			html.Div([
				html.H3('Dataset Output', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
				html.P('The output box now sits next to dataset handling so the dataset summary and preview update immediately beside the code cell.', style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45', 'marginBottom': '10px'}),
				html.Div(id='level3-dataset-output-box', style={'marginBottom': '12px', 'minHeight': '74px', 'padding': '12px', 'borderRadius': '12px', 'border': '1px solid #dbeafe', 'backgroundColor': '#f8fafc'}),
				html.Div(id='level3-dataset-summary', style={'marginBottom': '10px'}),
				dcc.Graph(id='level3-dataset-preview-graph', style={'height': '32vh'}),
			], style={**CARD_STYLE, 'flex': '1 1 460px', 'minWidth': '360px'}),
		], style={'display': 'flex', 'gap': '18px', 'flexWrap': 'wrap', 'alignItems': 'stretch', 'marginBottom': '18px'}),
		html.Div([
			html.Div([
				html.H3('Structure Code Cells', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
				html.P('Define the input layer, activation function, hidden stack, and output layer as separate code cells. The architecture panel on the right reflects the committed structure.', style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45', 'marginBottom': '10px'}),
				_level3_code_cell(2, '2. Input Layer', 'Set the feature-layer width used by the FNN.', 'Run Input Cell'),
				_level3_code_cell(3, '3. Activation Function', 'Choose the hidden-layer non-linearity.', 'Run Activation Cell'),
				_level3_code_cell(4, '4. Hidden Layers', 'Define the list of hidden-layer widths.', 'Run Hidden Layer Cell'),
				_level3_code_cell(5, '5. Output Layer', 'Define the output head used by the classifier.', 'Run Output Cell'),
				_level3_code_cell(6, '6. Training Configuration', 'Set the total epochs and learning rate that the cached playback will use.', 'Run Training Config Cell', code_height='110px'),
			], style={**CARD_STYLE, 'flex': '1 1 460px', 'minWidth': '360px'}),
			html.Div([
				html.H3('FNN Architecture', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
				html.P('This box now sits directly to the right of the structure cells. It updates from the latest committed code-defined architecture and animates once training playback starts.', style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45', 'marginBottom': '14px'}),
				dcc.Graph(id='level3-network-diagram-graph', style={'height': '54vh'}),
				html.Div(id='level3-arch-summary', style={'marginTop': '10px'}),
			], style={**CARD_STYLE, 'flex': '1.2 1 560px', 'minWidth': '420px'}),
		], style={'display': 'flex', 'gap': '18px', 'flexWrap': 'wrap', 'alignItems': 'stretch', 'marginBottom': '18px'}),
		html.Div([
			html.Div([
				html.H3('Model Control', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
				html.P('Training controls unlock after all six setup cells have been run. Training is cached as epoch stages so the animation can be paused and stepped.', style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45', 'marginBottom': '10px'}),
				html.Div([
					html.Button('Start Training', id='level3-train-toggle-btn', n_clicks=0, style={**CONTROL_BUTTON_STYLE, 'backgroundColor': '#0f766e', 'color': 'white'}),
					html.Button('Pause Training', id='level3-pause-btn', n_clicks=0, disabled=True, style={**CONTROL_BUTTON_STYLE, 'backgroundColor': '#b45309', 'color': 'white'}),
					html.Button('Reset Model', id='level3-reset-btn', n_clicks=0, style={**CONTROL_BUTTON_STYLE, 'backgroundColor': '#e2e8f0', 'color': '#0f172a'}),
				], style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '12px'}),
				html.Div('Reset returns the code-defined model to its untrained state while preserving the structure cells you already ran.', style={'fontSize': '11px', 'color': '#64748b', 'lineHeight': '1.4'}),
			], style={**CARD_STYLE, 'flex': '1 1 280px', 'minWidth': '260px'}),
			html.Div([
				html.H3('Animation Control', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
				html.P('Auto mode plays the cached stages continuously. Semi-auto mode lets you move stage by stage inside the current epoch.', style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45', 'marginBottom': '10px'}),
				html.Div([
					html.Label('Animation Speed', style={'display': 'block', 'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '6px'}),
					dcc.RadioItems(
						id='level3-play-speed-control',
						options=[
							{'label': 'Slow', 'value': 'slow'},
							{'label': 'Normal', 'value': 'normal'},
							{'label': 'Fast', 'value': 'fast'},
						],
						value='normal',
						inline=True,
						labelStyle={'marginRight': '16px', 'fontWeight': '600', 'color': '#334155'},
						inputStyle={'marginRight': '6px'},
					),
				], style={'marginBottom': '12px'}),
				html.Div([
					html.Label('Training Mode', style={'display': 'block', 'fontWeight': '600', 'fontSize': '13px', 'marginBottom': '6px'}),
					dcc.RadioItems(
						id='level3-training-mode',
						options=[
							{'label': 'Auto', 'value': 'auto'},
							{'label': 'Semi-Auto', 'value': 'semiauto'},
						],
						value='auto',
						inline=True,
						labelStyle={'marginRight': '16px', 'fontWeight': '600', 'color': '#334155'},
						inputStyle={'marginRight': '6px'},
					),
				], style={'marginBottom': '12px'}),
				html.Div([
					html.Button('Previous Stage', id='level3-prev-stage-btn', n_clicks=0, disabled=True, style={**CONTROL_BUTTON_STYLE, 'backgroundColor': '#475569', 'color': 'white'}),
					html.Button('Next Stage', id='level3-step-btn', n_clicks=0, disabled=True, style={**CONTROL_BUTTON_STYLE, 'backgroundColor': '#1d4ed8', 'color': 'white'}),
				], style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap'}),
			], style={**CARD_STYLE, 'flex': '1 1 320px', 'minWidth': '280px'}),
			html.Div(id='level3-output-summary', style={**CARD_STYLE, 'flex': '1 1 260px', 'minWidth': '240px', 'padding': '10px 12px'}),
			html.Div([
				html.H3('Training Cache', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
				html.P('Each completed epoch writes a compact log entry after the cached update stage has finished.', style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45', 'marginBottom': '10px'}),
				html.Div(id='level3-training-log'),
			], style={**CARD_STYLE, 'flex': '1 1 320px', 'minWidth': '280px'}),
		], style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'alignItems': 'stretch', 'marginBottom': '18px'}),
		html.Div([
			html.Div([
				html.H3('Decision Boundary', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
				html.P('Inspect the current classifier surface for the code-defined architecture.', style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45'}),
				dcc.Graph(id='level3-boundary-graph', style={'height': '42vh'}),
			], style={**CARD_STYLE, 'height': '100%'}),
			html.Div([
				html.H3('Training Curves', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
				html.P('The cached training run writes the same loss and accuracy history used by Level 2.', style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45'}),
				dcc.Graph(id='level3-loss-graph', style={'height': '42vh'}),
			], style={**CARD_STYLE, 'height': '100%'}),
			html.Div([
				html.H3('Activation Snapshot', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
				html.P('This view reflects the current hidden activations for a held-out batch using the displayed parameters.', style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45'}),
				dcc.Graph(id='level3-activations-graph', style={'height': '42vh'}),
			], style={**CARD_STYLE, 'height': '100%'}),
		], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(300px, 1fr))', 'gap': '16px'}),
		dcc.Interval(id='level3-train-interval', interval=350, n_intervals=0, disabled=True),
		dcc.Store(id='level3-params-store'),
		dcc.Store(id='level3-training-store'),
	], style=PAGE_STYLE)