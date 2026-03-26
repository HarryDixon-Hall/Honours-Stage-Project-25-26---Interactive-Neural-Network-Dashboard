from dash import dcc, html


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

LABEL_STYLE = {
    'display': 'block',
    'fontWeight': '600',
    'marginBottom': '6px',
    'color': '#1f2937',
    'fontSize': '13px',
}

INPUT_BOX_STYLE = {
    'marginBottom': '14px',
}

NUMBER_INPUT_STYLE = {
    'width': '100%',
    'padding': '8px 10px',
    'borderRadius': '10px',
    'border': '1px solid #cbd5e1',
    'fontSize': '13px',
    'boxSizing': 'border-box',
}


def level2_layout():
    return html.Div([
        html.H2(
            'Level 2 - Feed-Forward Neural Network Builder',
            style={'textAlign': 'center', 'marginBottom': '8px', 'fontSize': '28px'}
        ),
        html.P(
            'This level focuses on a binary classification workflow. Choose a dataset, shape the network architecture in the centre, control training on the left, and inspect performance and the learned decision boundary on the right.',
            style={
                'textAlign': 'center',
                'margin': '0 auto 18px auto',
                'maxWidth': '1040px',
                'color': '#475569',
                'lineHeight': '1.5',
                'fontSize': '14px',
            }
        ),
        html.Div([
            html.Div(
                id='level2-model-status',
                style={
                    'flex': '0.9 1 220px',
                    'padding': '12px 14px',
                    'borderRadius': '14px',
                    'backgroundColor': '#f8fafc',
                    'border': '1px solid #dbeafe',
                    'minWidth': '220px',
                }
            ),
            html.Div(
                id='level2-epoch-live',
                style={
                    'flex': '0.9 1 220px',
                    'padding': '12px 14px',
                    'borderRadius': '14px',
                    'backgroundColor': '#f8fafc',
                    'border': '1px solid #dbeafe',
                    'minWidth': '220px',
                }
            ),
            html.Div(
                id='level2-training-stage-panel',
                style={
                    'flex': '1.45 1 360px',
                    'padding': '12px 14px',
                    'borderRadius': '14px',
                    'backgroundColor': '#f8fafc',
                    'border': '1px solid #dbeafe',
                    'minWidth': '320px',
                }
            ),
        ], style={
            'display': 'flex',
            'gap': '12px',
            'flexWrap': 'wrap',
            'alignItems': 'stretch',
            'marginBottom': '16px',
        }),
        html.Div([
            html.Div([
                html.Div([
                    html.H3('Dataset Handling', style={'marginTop': '0', 'marginBottom': '6px', 'fontSize': '18px'}),
                    html.P(
                        'Choose the binary dataset used by the network. The raw inputs stay two-dimensional so the decision boundary can still be visualised directly.',
                        style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45', 'marginBottom': '8px'}
                    ),
                    html.Div('Classification Task', style={
                        'display': 'inline-block',
                        'padding': '3px 8px',
                        'borderRadius': '999px',
                        'backgroundColor': '#e0f2fe',
                        'color': '#0f172a',
                        'fontWeight': '600',
                        'fontSize': '12px',
                        'marginBottom': '10px',
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
                    ], style={'marginBottom': '10px'}),
                    html.Div([
                        html.Div('Available toy datasets', style={'fontWeight': '600', 'fontSize': '12px', 'marginBottom': '4px'}),
                        html.Ul([
                            html.Li('Moons: non-linear interleaving classes'),
                            html.Li('Circles: nested ring classification'),
                            html.Li('Linear: near linearly separable classes'),
                        ], style={'paddingLeft': '18px', 'marginBottom': '0', 'color': '#475569', 'lineHeight': '1.3', 'fontSize': '12px'}),
                    ]),
                ], style={**CARD_STYLE, 'padding': '16px 18px'}),
                html.Div([
                    html.Div([
                        html.H3('Model Control', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
                        html.P(
                            'Start, pause, and reset training here. This control is dominant over animation playback.',
                            style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45', 'marginBottom': '10px'}
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
                                    'padding': '10px 14px',
                                    'borderRadius': '10px',
                                    'fontWeight': '600',
                                    'fontSize': '13px',
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
                                    'padding': '10px 14px',
                                    'borderRadius': '10px',
                                    'fontWeight': '600',
                                    'fontSize': '13px',
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
                                    'padding': '10px 14px',
                                    'borderRadius': '10px',
                                    'fontWeight': '600',
                                    'fontSize': '13px',
                                    'cursor': 'pointer',
                                }
                            ),
                        ], style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '12px'}),
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
                        ], style={'marginBottom': '8px'}),
                        html.Div(
                            'Pausing freezes the current epoch stage. Reset rebuilds the model and returns the stage timeline to the start.',
                            style={'fontSize': '11px', 'color': '#64748b', 'lineHeight': '1.4'}
                        ),
                    ], style={**CARD_STYLE, 'flex': '1 1 280px', 'marginTop': '18px'}),
                    html.Div([
                        html.H3('Animation Control', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
                        html.P(
                            'Animation settings control how the current epoch is played back while training is active.',
                            style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45', 'marginBottom': '10px'}
                        ),
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
                        ], style={'marginBottom': '12px'}),
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
                        ], style={'marginBottom': '12px'}),
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
                                    'padding': '10px 14px',
                                    'borderRadius': '10px',
                                    'fontWeight': '600',
                                    'fontSize': '13px',
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
                                    'padding': '10px 14px',
                                    'borderRadius': '10px',
                                    'fontWeight': '600',
                                    'fontSize': '13px',
                                    'cursor': 'pointer',
                                }
                            ),
                        ], style={'display': 'flex', 'gap': '8px', 'flexWrap': 'wrap', 'marginBottom': '12px'}),
                        html.Div(
                            'Stage stepping is enabled only in semi-auto mode while the model is actively training. Auto mode locks stage stepping and uses the selected speed.',
                            style={'fontSize': '11px', 'color': '#64748b', 'lineHeight': '1.4'}
                        ),
                    ], style={**CARD_STYLE, 'flex': '1 1 280px', 'marginTop': '18px'}),
                ], style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap'}),
            ], style={'flex': '1 1 280px', 'minWidth': '280px', 'maxWidth': '360px'}),
            html.Div([
                html.Div([
                    html.H3('FNN Architecture', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
                    html.P(
                        'The network architecture is central in this level. Adjust the hidden-layer count, then edit each layer neuron count in the column aligned with that layer.',
                        style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45'}
                    ),
                    html.Div([
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
                        ], style={'flex': '1 1 420px'}),
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
                        ], style={'flex': '0.8 1 220px'}),
                    ], style={'display': 'flex', 'gap': '12px', 'flexWrap': 'wrap', 'marginBottom': '16px'}),
                    html.Div([
                        html.Div([
                            html.Div('Input Layer', style={'fontSize': '11px', 'textTransform': 'uppercase', 'letterSpacing': '0.08em', 'color': '#2563eb', 'fontWeight': '700', 'marginBottom': '6px'}),
                            dcc.Input(
                                id='level2-input-dim-input',
                                type='number',
                                min=2,
                                max=8,
                                step=1,
                                value=2,
                                style=NUMBER_INPUT_STYLE,
                            ),
                        ], style={'flex': '1 1 110px', 'minWidth': '110px'}),
                        html.Div([
                            html.Div('Hidden 1', style={'fontSize': '11px', 'textTransform': 'uppercase', 'letterSpacing': '0.08em', 'color': '#059669', 'fontWeight': '700', 'marginBottom': '6px'}),
                            dcc.Input(
                                id='level2-hidden-layer-1-input',
                                type='number',
                                min=2,
                                max=16,
                                step=1,
                                value=6,
                                style=NUMBER_INPUT_STYLE,
                            ),
                        ], id='level2-hidden-layer-1-wrapper', style={'flex': '1 1 110px', 'minWidth': '110px'}),
                        html.Div([
                            html.Div('Hidden 2', style={'fontSize': '11px', 'textTransform': 'uppercase', 'letterSpacing': '0.08em', 'color': '#059669', 'fontWeight': '700', 'marginBottom': '6px'}),
                            dcc.Input(
                                id='level2-hidden-layer-2-input',
                                type='number',
                                min=2,
                                max=16,
                                step=1,
                                value=6,
                                style=NUMBER_INPUT_STYLE,
                            ),
                        ], id='level2-hidden-layer-2-wrapper', style={'flex': '1 1 110px', 'minWidth': '110px'}),
                        html.Div([
                            html.Div('Hidden 3', style={'fontSize': '11px', 'textTransform': 'uppercase', 'letterSpacing': '0.08em', 'color': '#059669', 'fontWeight': '700', 'marginBottom': '6px'}),
                            dcc.Input(
                                id='level2-hidden-layer-3-input',
                                type='number',
                                min=2,
                                max=16,
                                step=1,
                                value=4,
                                style=NUMBER_INPUT_STYLE,
                            ),
                        ], id='level2-hidden-layer-3-wrapper', style={'flex': '1 1 110px', 'minWidth': '110px'}),
                        html.Div([
                            html.Div('Hidden 4', style={'fontSize': '11px', 'textTransform': 'uppercase', 'letterSpacing': '0.08em', 'color': '#059669', 'fontWeight': '700', 'marginBottom': '6px'}),
                            dcc.Input(
                                id='level2-hidden-layer-4-input',
                                type='number',
                                min=2,
                                max=16,
                                step=1,
                                value=4,
                                style=NUMBER_INPUT_STYLE,
                            ),
                        ], id='level2-hidden-layer-4-wrapper', style={'flex': '1 1 110px', 'minWidth': '110px'}),
                        html.Div([
                            html.Div('Output Layer', style={'fontSize': '11px', 'textTransform': 'uppercase', 'letterSpacing': '0.08em', 'color': '#dc2626', 'fontWeight': '700', 'marginBottom': '6px'}),
                            dcc.Input(
                                id='level2-output-dim-input',
                                type='number',
                                min=1,
                                max=2,
                                step=1,
                                value=1,
                                style=NUMBER_INPUT_STYLE,
                            ),
                        ], style={'flex': '1 1 110px', 'minWidth': '110px'}),
                    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'nowrap', 'overflowX': 'auto', 'alignItems': 'flex-start', 'marginBottom': '14px'}),
                    dcc.Graph(id='level2-network-diagram-graph', style={'height': '52vh'}),
                    html.Div(id='level2-math-explanation', style={'marginTop': '10px'}),
                ], style=CARD_STYLE),
            ], style={'flex': '1.55 1 620px', 'minWidth': '480px'}),
            html.Div([
                html.Div(
                    id='level2-output-summary',
                    style={
                        **CARD_STYLE,
                        'padding': '10px 12px',
                        'marginBottom': '14px',
                        'alignSelf': 'stretch',
                    }
                ),
                html.Div([
                    html.H3('Decision Boundary', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
                    html.P(
                        'Inspect the decision boundary learned for the selected dataset and compare it with the true sample locations.',
                        style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45'}
                    ),
                    dcc.Graph(id='level2-decision-boundary-graph', style={'height': '44vh'}),
                    html.Div(id='level2-boundary-explanation', style={'marginTop': '16px'}),
                ], style=CARD_STYLE),
                html.Div([
                    html.H3('Activation Function Visualisation', style={'marginTop': '0', 'marginBottom': '8px', 'fontSize': '18px'}),
                    html.P(
                        'This plot shows the non-linearity currently applied inside each hidden layer.',
                        style={'fontSize': '12px', 'color': '#475569', 'lineHeight': '1.45'}
                    ),
                    dcc.Graph(id='level2-activation-graph', style={'height': '28vh'}),
                ], style={**CARD_STYLE, 'marginTop': '14px'}),
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
