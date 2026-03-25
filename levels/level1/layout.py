from dash import dcc, html


def level1_layout():
    return html.Div(
        [
            html.H1(
                "Level 1: Preconfigured FFNN Explorer",
                style={
                    'textAlign': 'center',
                    'marginBottom': '8px'
                }
            ),
            html.P(
                "Work with a fixed feed-forward network and see how dataset choice, activation, and training settings change the learned boundary.",
                style={
                    'textAlign': 'center',
                    'maxWidth': '960px',
                    'margin': '0 auto 20px auto',
                    'color': '#4b5563'
                }
            ),
            html.Div(
                id='l1-metrics-cards',
                style={
                    'display': 'grid',
                    'gridTemplateColumns': 'repeat(auto-fit, minmax(150px, 1fr))',
                    'gap': '12px',
                    'marginBottom': '18px'
                }
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Control Panel", style={'marginTop': '0'}),
                            html.P(
                                "Level 1 keeps the model architecture fixed so the main focus stays on the ML pipeline: choose data, run training, and compare outcomes.",
                                style={'fontSize': '14px', 'color': '#4b5563'}
                            ),
                            html.Label("Dataset", style={'fontWeight': '600'}),
                            dcc.Dropdown(
                                id='l1-dataset',
                                options=[
                                    {'label': 'Linearly Separable', 'value': 'linear'},
                                    {'label': 'Moons', 'value': 'moons'},
                                    {'label': 'Circles', 'value': 'circles'}
                                ],
                                value='linear',
                                clearable=False,
                                style={'marginBottom': '14px'}
                            ),
                            html.Label("Hidden Activation", style={'fontWeight': '600'}),
                            dcc.Dropdown(
                                id='l1-activation',
                                options=[
                                    {'label': 'ReLU', 'value': 'relu'},
                                    {'label': 'Tanh', 'value': 'tanh'},
                                    {'label': 'Sigmoid', 'value': 'sigmoid'}
                                ],
                                value='tanh',
                                clearable=False,
                                style={'marginBottom': '14px'}
                            ),
                            html.Label("Epochs Per Run", style={'fontWeight': '600'}),
                            dcc.Slider(
                                id='l1-epochs',
                                min=10,
                                max=150,
                                step=10,
                                value=50,
                                marks={10: '10', 50: '50', 100: '100', 150: '150'},
                            ),
                            html.Details(
                                [
                                    html.Summary("Advanced Controls", style={'cursor': 'pointer', 'fontWeight': '600'}),
                                    html.Div(
                                        [
                                            html.Label("Learning Rate", style={'fontWeight': '600', 'marginTop': '12px'}),
                                            dcc.Slider(
                                                id='l1-lr',
                                                min=0.001,
                                                max=0.1,
                                                step=0.001,
                                                value=0.02,
                                                marks={0.001: '0.001', 0.02: '0.02', 0.05: '0.05', 0.1: '0.1'},
                                                tooltip={'placement': 'bottom', 'always_visible': True},
                                            ),
                                        ],
                                        style={'marginTop': '10px'}
                                    )
                                ],
                                style={'marginTop': '18px', 'marginBottom': '18px'}
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Run Training",
                                        id='l1-run-training',
                                        n_clicks=0,
                                        style={
                                            'backgroundColor': '#0f766e',
                                            'color': 'white',
                                            'border': 'none',
                                            'padding': '10px 16px',
                                            'borderRadius': '8px'
                                        }
                                    ),
                                    html.Button(
                                        "Reset",
                                        id='l1-reset',
                                        n_clicks=0,
                                        style={
                                            'marginLeft': '10px',
                                            'backgroundColor': '#e5e7eb',
                                            'border': 'none',
                                            'padding': '10px 16px',
                                            'borderRadius': '8px'
                                        }
                                    )
                                ],
                                style={'marginBottom': '18px'}
                            ),
                            html.Div(
                                id='l1-architecture-summary',
                                style={
                                    'backgroundColor': '#f8fafc',
                                    'border': '1px solid #dbeafe',
                                    'borderRadius': '12px',
                                    'padding': '16px',
                                    'marginBottom': '18px'
                                }
                            ),
                            html.Div(
                                [
                                    html.H4("Prediction / Sample Inspector", style={'marginTop': '0'}),
                                    html.Label("Sample Index", style={'fontWeight': '600'}),
                                    dcc.Slider(id='l1-sample-index', min=0, max=299, step=1, value=0),
                                    html.Div(
                                        id='l1-sample-inspector',
                                        style={
                                            'marginTop': '14px',
                                            'padding': '14px',
                                            'backgroundColor': '#fff7ed',
                                            'borderRadius': '12px',
                                            'border': '1px solid #fed7aa'
                                        }
                                    )
                                ],
                                style={
                                    'backgroundColor': 'white',
                                    'borderRadius': '12px',
                                    'padding': '16px',
                                    'boxShadow': '0 2px 8px rgba(15, 23, 42, 0.08)'
                                }
                            )
                        ],
                        style={
                            'flex': '0 0 32%',
                            'padding': '20px',
                            'backgroundColor': '#f8fafc',
                            'borderRadius': '16px',
                            'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)',
                            'boxSizing': 'border-box'
                        }
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Decision Boundary", style={'marginTop': '0'}),
                                    dcc.Graph(
                                        id='l1-decision-boundary',
                                        style={'height': '58vh'}
                                    )
                                ],
                                style={
                                    'backgroundColor': 'white',
                                    'borderRadius': '16px',
                                    'padding': '18px',
                                    'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)',
                                    'marginBottom': '16px'
                                }
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H3("Loss Curve", style={'marginTop': '0'}),
                                            dcc.Graph(
                                                id='l1-loss-curve',
                                                style={'height': '28vh'}
                                            )
                                        ],
                                        style={
                                            'flex': '1 1 60%',
                                            'backgroundColor': 'white',
                                            'borderRadius': '16px',
                                            'padding': '18px',
                                            'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)'
                                        }
                                    ),
                                    html.Div(
                                        [
                                            html.H3("Architecture View", style={'marginTop': '0'}),
                                            html.Div(id='l1-architecture-view')
                                        ],
                                        style={
                                            'flex': '1 1 40%',
                                            'backgroundColor': 'white',
                                            'borderRadius': '16px',
                                            'padding': '18px',
                                            'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)'
                                        }
                                    )
                                ],
                                style={
                                    'display': 'flex',
                                    'gap': '16px',
                                    'flexWrap': 'wrap'
                                }
                            )
                        ],
                        style={
                            'flex': '1',
                            'paddingLeft': '20px',
                            'boxSizing': 'border-box'
                        }
                    )
                ],
                style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'alignItems': 'flex-start'
                }
            ),
            dcc.Store(id='l1-training-store')
        ]
    )