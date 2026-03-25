from dash import dcc, html

from code_execution import CodeExecutionBox


LEVEL3_CELL_SNIPPETS = {
    1: (
        '# Cell 1: Load dataset\n'
        'dataset_name = "moons"\n'
        'X, y = load_toy_dataset(dataset_name)\n'
        'print(f"Dataset: {dataset_name}")\n'
        'print("X shape:", X.shape, "y shape:", y.shape)\n'
        'print("Class balance:", {0: int((y == 0).sum()), 1: int((y == 1).sum())})'
    ),
    2: (
        '# Cell 2: Define model\n'
        'hidden_layers = [6, 6]\n'
        'activation = "tanh"\n'
        'model = init_level2_mlp(input_dim=2, hidden_layers=hidden_layers, output_dim=1)\n'
        'print("Activation:", activation)\n'
        'print("Weight shapes:", [np.array(weight).shape for weight in model["weights"]])\n'
        'print("Bias shapes:", [np.array(bias).shape for bias in model["biases"]])'
    ),
    3: (
        '# Cell 3: Forward pass\n'
        'batch_X = X_train[:24]\n'
        'probs, cache = level2_forward_pass(batch_X, model, activation)\n'
        'print("Batch shape:", batch_X.shape)\n'
        'print("Output shape:", probs.shape)\n'
        'print("Hidden shapes:", [layer.shape for layer in cache["activations"][1:-1]])\n'
        'print("First 5 probabilities:", probs.flatten()[:5])'
    ),
    4: (
        '# Cell 4: Train model\n'
        'model = train_level2_model(X_train, y_train, model, activation=activation, epochs=epochs)\n'
        'metrics = level2_evaluate_metrics(X_test, y_test, model, activation, l2=1e-4)\n'
        'print("Epoch:", model["epoch"])\n'
        'print("Test accuracy:", round(metrics["accuracy"], 4))\n'
        'print("Test loss:", round(metrics["loss"], 4))'
    ),
    5: (
        '# Cell 5: Inspect internals\n'
        '_, cache = level2_forward_pass(X_test[:24], model, activation)\n'
        'print("Hidden layers:", len(cache["activations"]) - 2)\n'
        'print("Hidden activation shapes:", [layer.shape for layer in cache["activations"][1:-1]])'
    ),
    6: (
        '# Cell 6: Evaluate model\n'
        'probs, _ = level2_forward_pass(X_test, model, activation)\n'
        'pred_labels = (probs.flatten() >= 0.5).astype(int)\n'
        'print("Accuracy:", float((pred_labels == y_test).mean()))\n'
        'print("Confusion matrix:\n", confusion_matrix(y_test, pred_labels))'
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


def _level3_code_cell(cell_number, title, description, button_id, button_text, controls=None):
    if controls is None:
        controls = []

    _ = button_id
    return LEVEL3_CELL_EDITORS[cell_number].render(
        default_code=LEVEL3_CELL_SNIPPETS[cell_number],
        title=title,
        description=description,
        controls=controls,
        run_label=button_text,
        show_export=False,
        include_plot=False,
        code_height='160px',
        output_placeholder='Run the cell to see console output.',
        wrapper_style={'marginBottom': '16px'},
    )


def _level3_output_card(title, children):
    return html.Div(
        [
            html.H3(title, style={'marginTop': '0', 'marginBottom': '14px'}),
            children,
        ],
        style={
            'backgroundColor': 'white',
            'borderRadius': '16px',
            'padding': '18px',
            'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)',
            'height': '100%',
            'boxSizing': 'border-box',
        },
    )


def level3_layout():
    return html.Div(
        [
            html.H2(
                'Level 3 – Notebook Workflow for Building a Classifier',
                style={'textAlign': 'center', 'marginBottom': '12px'},
            ),
            html.P(
                'Use the same execution infrastructure as the sandbox while stepping through a structured classification workflow. '
                'Each cell stays editable, executes in a controlled environment, and still drives the guided visual outputs for this level.',
                style={
                    'textAlign': 'center',
                    'margin': '0 auto 24px auto',
                    'maxWidth': '980px',
                    'color': '#4b5563',
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                id='level3-notebook-status',
                                style={
                                    'backgroundColor': '#ecfeff',
                                    'border': '1px solid #a5f3fc',
                                    'borderRadius': '14px',
                                    'padding': '16px',
                                    'marginBottom': '16px',
                                    'fontSize': '13px',
                                    'color': '#155e75',
                                },
                            ),
                            _level3_code_cell(
                                1,
                                '1. Load Dataset',
                                'Choose a toy classification dataset and preview the split before building the model.',
                                'level3-load-data-btn',
                                'Run Cell 1',
                                controls=[
                                    html.Label('Toy classification dataset', style={'fontWeight': '600'}),
                                    dcc.Dropdown(
                                        id='level3-dataset-dropdown',
                                        options=[
                                            {'label': 'Moons', 'value': 'moons'},
                                            {'label': 'Circles', 'value': 'circles'},
                                            {'label': 'Linear', 'value': 'linear'},
                                        ],
                                        value='moons',
                                        clearable=False,
                                    ),
                                ],
                            ),
                            _level3_code_cell(
                                2,
                                '2. Define Model',
                                'Set the hidden stack and activation, then inspect the parameter shapes before training.',
                                'level3-define-model-btn',
                                'Run Cell 2',
                                controls=[
                                    html.Label('Hidden layers', style={'fontWeight': '600'}),
                                    dcc.Slider(
                                        id='level3-depth-slider',
                                        min=1,
                                        max=5,
                                        step=1,
                                        value=2,
                                        marks={index: str(index) for index in range(1, 6)},
                                    ),
                                    html.Label('Neurons per hidden layer', style={'fontWeight': '600', 'marginTop': '12px'}),
                                    dcc.Slider(
                                        id='level3-width-slider',
                                        min=2,
                                        max=12,
                                        step=1,
                                        value=6,
                                        marks={2: '2', 4: '4', 8: '8', 12: '12'},
                                    ),
                                    html.Label('Activation function', style={'fontWeight': '600', 'marginTop': '12px'}),
                                    dcc.Dropdown(
                                        id='level3-activation-dropdown',
                                        options=[
                                            {'label': 'ReLU', 'value': 'relu'},
                                            {'label': 'Tanh', 'value': 'tanh'},
                                            {'label': 'Sigmoid', 'value': 'sigmoid'},
                                        ],
                                        value='tanh',
                                        clearable=False,
                                    ),
                                ],
                            ),
                            _level3_code_cell(
                                3,
                                '3. Forward Pass',
                                'Push one batch through the network and inspect tensor shapes before training.',
                                'level3-forward-btn',
                                'Run Cell 3',
                            ),
                            _level3_code_cell(
                                4,
                                '4. Train Model',
                                'Optimise the classifier for a chosen number of epochs and inspect the resulting metrics.',
                                'level3-train-btn',
                                'Run Cell 4',
                                controls=[
                                    html.Label('Training epochs', style={'fontWeight': '600'}),
                                    dcc.Slider(
                                        id='level3-epochs-slider',
                                        min=50,
                                        max=400,
                                        step=50,
                                        value=150,
                                        marks={50: '50', 150: '150', 300: '300', 400: '400'},
                                    ),
                                ],
                            ),
                            _level3_code_cell(
                                5,
                                '5. Inspect Internals',
                                'Expose hidden representations and per-layer activations once the model has learned something useful.',
                                'level3-inspect-btn',
                                'Run Cell 5',
                            ),
                            _level3_code_cell(
                                6,
                                '6. Evaluate Model',
                                'Measure final predictive quality and inspect the confusion matrix directly from code.',
                                'level3-evaluate-btn',
                                'Run Cell 6',
                            ),
                        ],
                        style={
                            'flex': '0 0 38%',
                            'paddingRight': '20px',
                            'boxSizing': 'border-box',
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3('Decision Boundary / Prediction Surface', style={'marginTop': '0'}),
                                    dcc.Graph(id='level3-boundary-graph', style={'height': '52vh'}),
                                ],
                                style={
                                    'backgroundColor': 'white',
                                    'borderRadius': '16px',
                                    'padding': '18px',
                                    'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)',
                                    'marginBottom': '16px',
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H3('Loss Curve', style={'marginTop': '0'}),
                                            dcc.Graph(id='level3-loss-graph', style={'height': '30vh'}),
                                        ],
                                        style={
                                            'flex': '1 1 42%',
                                            'backgroundColor': 'white',
                                            'borderRadius': '16px',
                                            'padding': '18px',
                                            'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)',
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.H3('Per-Layer Activations', style={'marginTop': '0'}),
                                            dcc.Graph(id='level3-activations-graph', style={'height': '30vh'}),
                                        ],
                                        style={
                                            'flex': '1 1 58%',
                                            'backgroundColor': 'white',
                                            'borderRadius': '16px',
                                            'padding': '18px',
                                            'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)',
                                        },
                                    ),
                                ],
                                style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'},
                            ),
                        ],
                        style={'flex': '1', 'boxSizing': 'border-box'},
                    ),
                ],
                style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'alignItems': 'flex-start',
                    'marginBottom': '22px',
                },
            ),
            html.H3('Cell Outputs', style={'marginBottom': '14px'}),
            html.Div(
                [
                    html.Div(
                        [
                            _level3_output_card(
                                'Cell 1 Output – Dataset Preview',
                                html.Div([
                                    dcc.Graph(id='level3-dataset-preview-graph', style={'height': '30vh'}),
                                    html.Div(id='level3-dataset-summary'),
                                ]),
                            )
                        ],
                        style={'flex': '1 1 48%', 'minWidth': '320px'},
                    ),
                    html.Div(
                        [
                            _level3_output_card(
                                'Cell 2 Output – Architecture',
                                html.Div([
                                    dcc.Graph(id='level3-network-diagram-graph', style={'height': '30vh'}),
                                    html.Div(id='level3-arch-summary'),
                                ]),
                            )
                        ],
                        style={'flex': '1 1 48%', 'minWidth': '320px'},
                    ),
                    html.Div([
                        _level3_output_card('Cell 3 Output – Forward Pass', html.Div(id='level3-forward-output'))
                    ], style={'flex': '1 1 48%', 'minWidth': '320px'}),
                    html.Div([
                        _level3_output_card('Cell 4 Output – Training Log', html.Div(id='level3-training-log'))
                    ], style={'flex': '1 1 48%', 'minWidth': '320px'}),
                    html.Div([
                        _level3_output_card(
                            'Cell 5 Output – Hidden Representations',
                            dcc.Graph(id='level3-hidden-space-graph', style={'height': '34vh'}),
                        )
                    ], style={'flex': '1 1 48%', 'minWidth': '320px'}),
                    html.Div([
                        _level3_output_card(
                            'Cell 6 Output – Evaluation',
                            html.Div([
                                dcc.Graph(id='level3-confusion-matrix-graph', style={'height': '28vh'}),
                                dcc.Graph(id='level3-misclassified-graph', style={'height': '28vh'}),
                                html.Div(id='level3-metrics-summary'),
                            ]),
                        )
                    ], style={'flex': '1 1 48%', 'minWidth': '320px'}),
                ],
                style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '16px'},
            ),
            dcc.Store(id='level3-params-store'),
        ]
    )
