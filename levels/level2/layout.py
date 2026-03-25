from dash import html, dcc

def level2_layout():
    return html.Div([
        html.H2(
            "Level 2 – Model Builder and Training Pipeline",
            style={'textAlign': 'center', 'marginBottom': '12px'}
        ),

        html.P(
            "Build a small feed-forward classifier from the same ideas introduced in Level 1. "
            "Choose the dataset, activation, and architecture, then train and compare runs to "
            "see how design choices change the learned boundary and optimisation behaviour.",
            style={
                'textAlign': 'center',
                'margin': '0 auto 24px auto',
                'maxWidth': '960px',
                'color': '#4b5563'
            }
        ),

        html.Div(
            id='level2-metrics-row',
            style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(170px, 1fr))',
                'gap': '12px',
                'marginBottom': '18px'
            }
        ),

        html.Div([
            html.Div([
                html.H3("Builder Controls", style={'marginTop': '0'}),
                html.P(
                    "Adjust the model architecture before training. Any control change resets the current run so the diagram and metrics stay in sync with the chosen design.",
                    style={'fontSize': '14px', 'color': '#4b5563'}
                ),

                html.Label("Hidden layers", style={'fontWeight': '600'}),
                dcc.Slider(
                    id='level2-hidden-layers-slider',
                    min=1,
                    max=4,
                    step=1,
                    value=2,
                    marks={i: str(i) for i in range(1, 5)}
                ),

                html.Label("Neurons per hidden layer", style={'fontWeight': '600', 'marginTop': '16px'}),
                dcc.Slider(
                    id='level2-neurons-slider',
                    min=2,
                    max=12,
                    step=1,
                    value=6,
                    marks={2: '2', 4: '4', 8: '8', 12: '12'}
                ),

                html.Label("Activation function", style={'fontWeight': '600', 'marginTop': '16px'}),
                dcc.Dropdown(
                    id='level2-activation-dropdown',
                    options=[
                        {'label': 'ReLU', 'value': 'relu'},
                        {'label': 'Tanh', 'value': 'tanh'},
                        {'label': 'Sigmoid', 'value': 'sigmoid'},
                    ],
                    value='tanh',
                    clearable=False,
                    style={'marginBottom': '14px'}
                ),

                html.Label("Toy dataset", style={'fontWeight': '600'}),
                dcc.Dropdown(
                    id='level2-dataset-dropdown',
                    options=[
                        {'label': 'Moons', 'value': 'moons'},
                        {'label': 'Circles', 'value': 'circles'},
                        {'label': 'Linear', 'value': 'linear'},
                    ],
                    value='moons',
                    clearable=False,
                    style={'marginBottom': '18px'}
                ),

                html.Div([
                    html.Button(
                        'Train Model',
                        id='level2-train-btn',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#0f766e',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 16px',
                            'borderRadius': '8px',
                            'marginRight': '8px'
                        }
                    ),
                    html.Button(
                        'Reset',
                        id='level2-reset-btn',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#e5e7eb',
                            'border': 'none',
                            'padding': '10px 16px',
                            'borderRadius': '8px',
                            'marginRight': '8px'
                        }
                    ),
                    html.Button(
                        'Compare Run',
                        id='level2-compare-btn',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#dbeafe',
                            'border': 'none',
                            'padding': '10px 16px',
                            'borderRadius': '8px'
                        }
                    ),
                ], style={'marginBottom': '18px'}),

                html.Div(
                    id='level2-comparison-panel',
                    style={
                        'backgroundColor': '#ffffff',
                        'border': '1px solid #dbeafe',
                        'borderRadius': '14px',
                        'padding': '16px',
                        'boxShadow': '0 2px 8px rgba(15, 23, 42, 0.05)'
                    }
                ),
            ], style={
                'flex': '0 0 30%',
                'padding': '20px',
                'backgroundColor': '#f8fafc',
                'borderRadius': '16px',
                'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)',
                'boxSizing': 'border-box'
            }),

            html.Div([
                html.Div([
                    html.H3("Decision Boundary", style={'marginTop': '0'}),
                    dcc.Graph(id='level2-decision-boundary-graph', style={'height': '52vh'})
                ], style={
                    'backgroundColor': 'white',
                    'borderRadius': '16px',
                    'padding': '18px',
                    'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)',
                    'marginBottom': '16px'
                }),

                html.Div([
                    html.Div([
                        html.H3("Training Curves", style={'marginTop': '0'}),
                        dcc.Graph(id='level2-training-curves-graph', style={'height': '30vh'})
                    ], style={
                        'flex': '1 1 52%',
                        'backgroundColor': 'white',
                        'borderRadius': '16px',
                        'padding': '18px',
                        'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)'
                    }),

                    html.Div([
                        html.H3("Activation Function", style={'marginTop': '0'}),
                        dcc.Graph(id='level2-activation-graph', style={'height': '30vh'})
                    ], style={
                        'flex': '1 1 48%',
                        'backgroundColor': 'white',
                        'borderRadius': '16px',
                        'padding': '18px',
                        'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)'
                    }),
                ], style={
                    'display': 'flex',
                    'gap': '16px',
                    'flexWrap': 'wrap',
                    'marginBottom': '16px'
                }),

                html.Div([
                    html.Div([
                        html.H3("Architecture Diagram", style={'marginTop': '0'}),
                        dcc.Graph(id='level2-network-diagram-graph', style={'height': '38vh'})
                    ], style={
                        'flex': '1 1 60%',
                        'backgroundColor': 'white',
                        'borderRadius': '16px',
                        'padding': '18px',
                        'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)'
                    }),

                    html.Div([
                        html.H3("Model Summary", style={'marginTop': '0'}),
                        html.Div(id='level2-math-explanation')
                    ], style={
                        'flex': '1 1 40%',
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
                'paddingLeft': '20px',
                'boxSizing': 'border-box'
            }),
        ], style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'alignItems': 'flex-start'
        }),

        # Stores for model state and saved comparison snapshot
        dcc.Store(id='level2-params-store'),
        dcc.Store(id='level2-compare-store'),
    ])
