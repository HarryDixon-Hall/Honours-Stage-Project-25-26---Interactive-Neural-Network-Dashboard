from dash import dcc, html


PAGE_STYLE = {
    'padding': '8px 0 24px 0',
}

CARD_STYLE = {
    'backgroundColor': '#ffffff',
    'border': '1px solid #d7e3f4',
    'borderRadius': '18px',
    'padding': '20px',
    'boxShadow': '0 14px 34px rgba(15, 23, 42, 0.08)',
    'boxSizing': 'border-box',
}

LABEL_STYLE = {
    'display': 'block',
    'fontWeight': '600',
    'marginBottom': '8px',
    'color': '#1f2937',
}

INPUT_BOX_STYLE = {
    'marginBottom': '18px',
}

NUMBER_INPUT_STYLE = {
    'width': '100%',
    'padding': '10px 12px',
    'borderRadius': '10px',
    'border': '1px solid #cbd5e1',
    'fontSize': '14px',
    'boxSizing': 'border-box',
}


def level2_layout():
    return html.Div([
        html.H2(
            'Level 2 - Feed-Forward Neural Network Builder',
            style={'textAlign': 'center', 'marginBottom': '10px'}
        ),
        html.P(
            'This level focuses on a binary classification workflow. Choose a dataset, shape the network architecture in the centre, control training on the left, and inspect performance and the learned decision boundary on the right.',
            style={
                'textAlign': 'center',
                'margin': '0 auto 24px auto',
                'maxWidth': '1040px',
                'color': '#475569',
                'lineHeight': '1.6',
            }
        ),
        html.Div([
            html.Div([
                html.Div([
                    html.H3('Dataset Handling', style={'marginTop': '0', 'marginBottom': '10px'}),
                    html.P(
                        'This box controls the binary classification dataset used by the network. The inputs remain two-dimensional in raw space so the decision boundary can be visualised directly.',
                        style={'fontSize': '14px', 'color': '#475569', 'lineHeight': '1.6'}
                    ),
                    html.Div('Classification Task', style={
                        'display': 'inline-block',
                        'padding': '6px 10px',
                        'borderRadius': '999px',
                        'backgroundColor': '#e0f2fe',
                        'color': '#0f172a',
                        'fontWeight': '600',
                        'marginBottom': '16px',
                    }),
                    html.Div([
                        html.Label('Dataset', style=LABEL_STYLE),
                        dcc.Dropdown(
                            id='level2-dataset-dropdown',
                            options=[
                                {'label': 'Moons', 'value': 'moons'},
                                {'label': 'Circles', 'value': 'circles'},
                                {'label': 'Linear', 'value': 'linear'},
                            ],
                            value='moons',
                            clearable=False,
                        ),
                    ], style=INPUT_BOX_STYLE),
                    html.Div([
                        html.Div('Available toy datasets', style={'fontWeight': '600', 'marginBottom': '8px'}),
                        html.Ul([
                            html.Li('Moons: non-linear interleaving classes'),
                            html.Li('Circles: nested ring classification'),
                            html.Li('Linear: near linearly separable classes'),
                        ], style={'paddingLeft': '18px', 'marginBottom': '0', 'color': '#475569'}),
                    ]),
                ], style=CARD_STYLE),
                html.Div([
                    html.H3('Training Control', style={'marginTop': '0', 'marginBottom': '10px'}),
                    html.P(
                        'Start, pause, or manually step optimisation, monitor the live epoch count, and tune the learning rate used for gradient descent.',
                        style={'fontSize': '14px', 'color': '#475569', 'lineHeight': '1.6'}
                    ),
                    html.Div([
                        html.Button(
                            'Start Training',
                            id='level2-train-toggle-btn',
                            n_clicks=0,
                            style={
                                'backgroundColor': '#0f766e',
                                'color': 'white',
                                'border': 'none',
                                'padding': '12px 16px',
                                'borderRadius': '10px',
                                'fontWeight': '600',
                                'cursor': 'pointer',
                            }
                        ),
                        html.Button(
                            'Pause Training',
                            id='level2-pause-btn',
                            n_clicks=0,
                            disabled=True,
                            style={
                                'backgroundColor': '#b45309',
                                'color': 'white',
                                'border': 'none',
                                'padding': '12px 16px',
                                'borderRadius': '10px',
                                'fontWeight': '600',
                                'cursor': 'pointer',
                            }
                        ),
                        html.Button(
                            'Reset Model',
                            id='level2-reset-btn',
                            n_clicks=0,
                            style={
                                'backgroundColor': '#e2e8f0',
                                'color': '#0f172a',
                                'border': 'none',
                                'padding': '12px 16px',
                                'borderRadius': '10px',
                                'fontWeight': '600',
                                'cursor': 'pointer',
                            }
                        ),
                    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'marginBottom': '18px'}),
                    html.Div([
                        html.Label('Training Mode', style=LABEL_STYLE),
                        dcc.RadioItems(
                            id='level2-training-mode',
                            options=[
                                {'label': 'Auto', 'value': 'auto'},
                                {'label': 'Semi-Auto', 'value': 'semiauto'},
                            ],
                            value='auto',
                            inline=True,
                            labelStyle={'marginRight': '16px', 'fontWeight': '600', 'color': '#334155'},
                            inputStyle={'marginRight': '6px'},
                        ),
                    ], style={'marginBottom': '18px'}),
                    html.Div([
                        html.Button(
                            'Previous Stage',
                            id='level2-prev-stage-btn',
                            n_clicks=0,
                            disabled=True,
                            style={
                                'backgroundColor': '#475569',
                                'color': 'white',
                                'border': 'none',
                                'padding': '12px 16px',
                                'borderRadius': '10px',
                                'fontWeight': '600',
                                'cursor': 'pointer',
                            }
                        ),
                        html.Button(
                            'Next Stage',
                            id='level2-step-btn',
                            n_clicks=0,
                            disabled=True,
                            style={
                                'backgroundColor': '#1d4ed8',
                                'color': 'white',
                                'border': 'none',
                                'padding': '12px 16px',
                                'borderRadius': '10px',
                                'fontWeight': '600',
                                'cursor': 'pointer',
                            }
                        ),
                    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'marginBottom': '18px'}),
                    html.Div(
                        id='level2-epoch-live',
                        style={
                            'display': 'flex',
                            'justifyContent': 'space-between',
                            'alignItems': 'center',
                            'padding': '14px 16px',
                            'borderRadius': '14px',
                            'backgroundColor': '#f8fafc',
                            'border': '1px solid #dbeafe',
                            'marginBottom': '18px',
                        }
                    ),
                    html.Div(
                        id='level2-training-stage-panel',
                        style={
                            'marginBottom': '18px',
                        }
                    ),
                    html.Div([
                        html.Label('Learning Rate', style=LABEL_STYLE),
                        dcc.Slider(
                            id='level2-learning-rate-slider',
                            min=0.01,
                            max=0.2,
                            step=0.01,
                            value=0.08,
                            marks={0.01: '0.01', 0.05: '0.05', 0.1: '0.10', 0.15: '0.15', 0.2: '0.20'},
                        ),
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Label('Animation Speed', style=LABEL_STYLE),
                        dcc.RadioItems(
                            id='level2-play-speed-control',
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
                    ], style={'marginBottom': '10px'}),
                    html.Div(
                        'A higher learning rate moves faster but can overshoot the best boundary. A lower rate is steadier but slower.',
                        style={'fontSize': '12px', 'color': '#64748b', 'lineHeight': '1.5'}
                    ),
                ], style={**CARD_STYLE, 'marginTop': '18px'}),
            ], style={'flex': '1 1 280px', 'minWidth': '280px'}),
            html.Div([
                html.Div([
                    html.H3('FNN Architecture', style={'marginTop': '0', 'marginBottom': '10px'}),
                    html.P(
                        'The network architecture is the centre of this level. Adjust the feature layer width, hidden-layer count, neuron count per hidden layer, output neurons, and activation function here.',
                        style={'fontSize': '14px', 'color': '#475569', 'lineHeight': '1.6'}
                    ),
                    html.Div([
                        html.Div([
                            html.Label('Feature Layer Neurons', style=LABEL_STYLE),
                            dcc.Input(
                                id='level2-input-dim-input',
                                type='number',
                                min=2,
                                max=8,
                                step=1,
                                value=2,
                                style=NUMBER_INPUT_STYLE,
                            ),
                        ], style={'flex': '1 1 180px'}),
                        html.Div([
                            html.Label('Output Layer Neurons', style=LABEL_STYLE),
                            dcc.Input(
                                id='level2-output-dim-input',
                                type='number',
                                min=1,
                                max=2,
                                step=1,
                                value=1,
                                style=NUMBER_INPUT_STYLE,
                            ),
                        ], style={'flex': '1 1 180px'}),
                        html.Div([
                            html.Label('Activation Function', style=LABEL_STYLE),
                            dcc.Dropdown(
                                id='level2-activation-dropdown',
                                options=[
                                    {'label': 'ReLU', 'value': 'relu'},
                                    {'label': 'Tanh', 'value': 'tanh'},
                                    {'label': 'Sigmoid', 'value': 'sigmoid'},
                                ],
                                value='tanh',
                                clearable=False,
                            ),
                        ], style={'flex': '1 1 220px'}),
                    ], style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap', 'marginBottom': '20px'}),
                    html.Div([
                        html.Label('Hidden Layer Count', style=LABEL_STYLE),
                        dcc.Slider(
                            id='level2-hidden-layers-slider',
                            min=1,
                            max=4,
                            step=1,
                            value=2,
                            marks={i: str(i) for i in range(1, 5)},
                        ),
                    ], style={'marginBottom': '20px'}),
                    html.Div([
                        html.Div([
                            html.Label('Hidden Layer 1 Neurons', style=LABEL_STYLE),
                            dcc.Input(
                                id='level2-hidden-layer-1-input',
                                type='number',
                                min=2,
                                max=16,
                                step=1,
                                value=6,
                                style=NUMBER_INPUT_STYLE,
                            ),
                        ], id='level2-hidden-layer-1-wrapper', style={'flex': '1 1 200px'}),
                        html.Div([
                            html.Label('Hidden Layer 2 Neurons', style=LABEL_STYLE),
                            dcc.Input(
                                id='level2-hidden-layer-2-input',
                                type='number',
                                min=2,
                                max=16,
                                step=1,
                                value=6,
                                style=NUMBER_INPUT_STYLE,
                            ),
                        ], id='level2-hidden-layer-2-wrapper', style={'flex': '1 1 200px'}),
                        html.Div([
                            html.Label('Hidden Layer 3 Neurons', style=LABEL_STYLE),
                            dcc.Input(
                                id='level2-hidden-layer-3-input',
                                type='number',
                                min=2,
                                max=16,
                                step=1,
                                value=4,
                                style=NUMBER_INPUT_STYLE,
                            ),
                        ], id='level2-hidden-layer-3-wrapper', style={'flex': '1 1 200px'}),
                        html.Div([
                            html.Label('Hidden Layer 4 Neurons', style=LABEL_STYLE),
                            dcc.Input(
                                id='level2-hidden-layer-4-input',
                                type='number',
                                min=2,
                                max=16,
                                step=1,
                                value=4,
                                style=NUMBER_INPUT_STYLE,
                            ),
                        ], id='level2-hidden-layer-4-wrapper', style={'flex': '1 1 200px'}),
                    ], style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap', 'marginBottom': '22px'}),
                    dcc.Graph(id='level2-network-diagram-graph', style={'height': '48vh'}),
                    html.Div(id='level2-math-explanation', style={'marginTop': '10px'}),
                ], style=CARD_STYLE),
            ], style={'flex': '1.35 1 520px', 'minWidth': '340px'}),
            html.Div([
                html.Div([
                    html.H3('Model Output', style={'marginTop': '0', 'marginBottom': '10px'}),
                    html.P(
                        'Inspect train and test metrics alongside the decision boundary learned for the selected dataset.',
                        style={'fontSize': '14px', 'color': '#475569', 'lineHeight': '1.6'}
                    ),
                    html.Div(id='level2-output-summary', style={'marginBottom': '18px'}),
                    dcc.Graph(id='level2-decision-boundary-graph', style={'height': '44vh'}),
                    html.Div(id='level2-boundary-explanation', style={'marginTop': '16px'}),
                ], style=CARD_STYLE),
                html.Div([
                    html.H3('Activation Function Visualisation', style={'marginTop': '0', 'marginBottom': '10px'}),
                    html.P(
                        'This plot shows the non-linearity currently applied inside each hidden layer.',
                        style={'fontSize': '14px', 'color': '#475569', 'lineHeight': '1.6'}
                    ),
                    dcc.Graph(id='level2-activation-graph', style={'height': '28vh'}),
                ], style={**CARD_STYLE, 'marginTop': '18px'}),
            ], style={'flex': '1 1 320px', 'minWidth': '300px'}),
        ], style={
            'display': 'flex',
            'gap': '20px',
            'alignItems': 'flex-start',
            'flexWrap': 'wrap',
        }),
        dcc.Interval(id='level2-train-interval', interval=350, n_intervals=0, disabled=True),
        dcc.Store(id='level2-params-store'),
        dcc.Store(id='level2-training-store', data={'running': False, 'paused': False, 'mode': 'auto'}),
    ], style=PAGE_STYLE)
