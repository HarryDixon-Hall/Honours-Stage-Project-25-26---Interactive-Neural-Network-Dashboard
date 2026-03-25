from dash import dcc, html

#3. level 3 - building a model with multiple code cells (visualisation of intermediate outputs)
def level3_layout():
    return html.Div([
        html.H2(
            "Level 3 – Notebook Workflow for Building a Classifier",
            style={'textAlign': 'center', 'marginBottom': '12px'}
        ),

        html.P(
            "Move from dashboard controls into a notebook-like workflow. Each cell defines one part "
            "of the classification pipeline, and every step exposes an intermediate output so you can "
            "inspect the dataset, architecture, forward pass, training behaviour, hidden states, and evaluation.",
            style={
                'textAlign': 'center',
                'margin': '0 auto 24px auto',
                'maxWidth': '980px',
                'color': '#4b5563'
            }
        ),

        html.Div([
            html.Div([
                html.Div(
                    id='level3-notebook-status',
                    style={
                        'backgroundColor': '#ecfeff',
                        'border': '1px solid #a5f3fc',
                        'borderRadius': '14px',
                        'padding': '16px',
                        'marginBottom': '16px',
                        'fontSize': '13px',
                        'color': '#155e75'
                    }
                ),
                _level3_code_cell(
                    "1. Load Dataset",
                    "Choose a toy classification dataset and preview the split before building the model.",
                    "# Cell 1: Load dataset\n"
                    "dataset_name = \"moons\"\n"
                    "X, y = load_toy_dataset(dataset_name)\n"
                    "X_train, X_test, y_train, y_test = train_test_split(\n"
                    "    X, y, test_size=0.25, random_state=42, stratify=y\n"
                    ")\n"
                    "preview_dataset(X_train, X_test, y_train, y_test)",
                    'level3-load-data-btn',
                    'Run Cell 1',
                    controls=[
                        html.Label("Toy classification dataset", style={'fontWeight': '600'}),
                        dcc.Dropdown(
                            id='level3-dataset-dropdown',
                            options=[
                                {'label': 'Moons', 'value': 'moons'},
                                {'label': 'Circles', 'value': 'circles'},
                                {'label': 'Linear', 'value': 'linear'},
                            ],
                            value='moons',
                            clearable=False
                        )
                    ]
                ),
                _level3_code_cell(
                    "2. Define Model",
                    "Set the hidden stack and activation inside the model-definition cell rather than using the whole page as a slider demo.",
                    "# Cell 2: Define model\n"
                    "hidden_layers = [6, 6]\n"
                    "activation = \"tanh\"\n"
                    "model = init_level2_mlp(\n"
                    "    input_dim=2, hidden_layers=hidden_layers, output_dim=1\n"
                    ")\n"
                    "summarise_architecture(model, activation)",
                    'level3-define-model-btn',
                    'Run Cell 2',
                    controls=[
                        html.Label("Hidden layers", style={'fontWeight': '600'}),
                        dcc.Slider(
                            id='level3-depth-slider',
                            min=1, max=5, step=1, value=2,
                            marks={i: str(i) for i in range(1, 6)}
                        ),
                        html.Label("Neurons per hidden layer", style={'fontWeight': '600', 'marginTop': '12px'}),
                        dcc.Slider(
                            id='level3-width-slider',
                            min=2, max=12, step=1, value=6,
                            marks={2: '2', 4: '4', 8: '8', 12: '12'}
                        ),
                        html.Label("Activation function", style={'fontWeight': '600', 'marginTop': '12px'}),
                        dcc.Dropdown(
                            id='level3-activation-dropdown',
                            options=[
                                {'label': 'ReLU', 'value': 'relu'},
                                {'label': 'Tanh', 'value': 'tanh'},
                                {'label': 'Sigmoid', 'value': 'sigmoid'},
                            ],
                            value='tanh',
                            clearable=False
                        )
                    ]
                ),
                _level3_code_cell(
                    "3. Forward Pass",
                    "Push one batch through the network and inspect tensor shapes and sample predictions before training.",
                    "# Cell 3: Forward pass\n"
                    "batch_X = X_train[:24]\n"
                    "probs, cache = level2_forward_pass(batch_X, model, activation)\n"
                    "print_shapes(cache)\n"
                    "show_example_predictions(batch_X, probs)",
                    'level3-forward-btn',
                    'Run Cell 3'
                ),
                _level3_code_cell(
                    "4. Train Model",
                    "Optimise the classifier for a chosen number of epochs and inspect how the boundary and loss change.",
                    "# Cell 4: Train model\n"
                    "epochs = 150\n"
                    "model = train_level2_model(\n"
                    "    X_train, y_train, model, activation=activation, epochs=epochs\n"
                    ")\n"
                    "plot_training_history(model['history'])",
                    'level3-train-btn',
                    'Run Cell 4',
                    controls=[
                        html.Label("Training epochs", style={'fontWeight': '600'}),
                        dcc.Slider(
                            id='level3-epochs-slider',
                            min=50, max=400, step=50, value=150,
                            marks={50: '50', 150: '150', 300: '300', 400: '400'}
                        )
                    ]
                ),
                _level3_code_cell(
                    "5. Inspect Internals",
                    "Expose hidden representations and per-layer activations once the model has meaningful state.",
                    "# Cell 5: Inspect internals\n"
                    "hidden_states = inspect_hidden_layers(model, X_test)\n"
                    "plot_activation_heatmaps(hidden_states)\n"
                    "plot_hidden_projection(hidden_states[-1], y_test)",
                    'level3-inspect-btn',
                    'Run Cell 5'
                ),
                _level3_code_cell(
                    "6. Evaluate Model",
                    "Measure final predictive quality and inspect where the network still makes mistakes.",
                    "# Cell 6: Evaluate model\n"
                    "metrics = evaluate_classifier(model, X_test, y_test)\n"
                    "plot_confusion_matrix(metrics['confusion_matrix'])\n"
                    "highlight_misclassified_points(X_test, y_test, metrics)",
                    'level3-evaluate-btn',
                    'Run Cell 6'
                ),
            ], style={
                'flex': '0 0 38%',
                'paddingRight': '20px',
                'boxSizing': 'border-box'
            }),

            html.Div([
                html.Div([
                    html.H3("Decision Boundary / Prediction Surface", style={'marginTop': '0'}),
                    dcc.Graph(id='level3-boundary-graph', style={'height': '52vh'})
                ], style={
                    'backgroundColor': 'white',
                    'borderRadius': '16px',
                    'padding': '18px',
                    'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)',
                    'marginBottom': '16px'
                }),
                html.Div([
                    html.Div([
                        html.H3("Loss Curve", style={'marginTop': '0'}),
                        dcc.Graph(id='level3-loss-graph', style={'height': '30vh'})
                    ], style={
                        'flex': '1 1 42%',
                        'backgroundColor': 'white',
                        'borderRadius': '16px',
                        'padding': '18px',
                        'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)'
                    }),
                    html.Div([
                        html.H3("Per-Layer Activations", style={'marginTop': '0'}),
                        dcc.Graph(id='level3-activations-graph', style={'height': '30vh'})
                    ], style={
                        'flex': '1 1 58%',
                        'backgroundColor': 'white',
                        'borderRadius': '16px',
                        'padding': '18px',
                        'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)'
                    }),
                ], style={
                    'display': 'flex',
                    'gap': '16px',
                    'flexWrap': 'wrap'
                }),
            ], style={
                'flex': '1',
                'boxSizing': 'border-box'
            }),
        ], style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'alignItems': 'flex-start',
            'marginBottom': '22px'
        }),

        html.H3("Cell Outputs", style={'marginBottom': '14px'}),
        html.Div([
            html.Div([
                _level3_output_card(
                    "Cell 1 Output – Dataset Preview",
                    html.Div([
                        dcc.Graph(id='level3-dataset-preview-graph', style={'height': '30vh'}),
                        html.Div(id='level3-dataset-summary')
                    ])
                )
            ], style={'flex': '1 1 48%', 'minWidth': '320px'}),
            html.Div([
                _level3_output_card(
                    "Cell 2 Output – Architecture",
                    html.Div([
                        dcc.Graph(id='level3-network-diagram-graph', style={'height': '30vh'}),
                        html.Div(id='level3-arch-summary')
                    ])
                )
            ], style={'flex': '1 1 48%', 'minWidth': '320px'}),
            html.Div([
                _level3_output_card(
                    "Cell 3 Output – Forward Pass",
                    html.Div(id='level3-forward-output')
                )
            ], style={'flex': '1 1 48%', 'minWidth': '320px'}),
            html.Div([
                _level3_output_card(
                    "Cell 4 Output – Training Log",
                    html.Div(id='level3-training-log')
                )
            ], style={'flex': '1 1 48%', 'minWidth': '320px'}),
            html.Div([
                _level3_output_card(
                    "Cell 5 Output – Hidden Representations",
                    dcc.Graph(id='level3-hidden-space-graph', style={'height': '34vh'})
                )
            ], style={'flex': '1 1 48%', 'minWidth': '320px'}),
            html.Div([
                _level3_output_card(
                    "Cell 6 Output – Evaluation",
                    html.Div([
                        dcc.Graph(id='level3-confusion-matrix-graph', style={'height': '28vh'}),
                        dcc.Graph(id='level3-misclassified-graph', style={'height': '28vh'}),
                        html.Div(id='level3-metrics-summary')
                    ])
                )
            ], style={'flex': '1 1 48%', 'minWidth': '320px'}),
        ], style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'gap': '16px'
        }),

        dcc.Store(id='level3-params-store'),
    ])
#endregion