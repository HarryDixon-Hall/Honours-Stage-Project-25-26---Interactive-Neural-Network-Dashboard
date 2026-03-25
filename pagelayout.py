#DECLARATION: CODE HERE IS ASSISTED BY https://www.perplexity.ai/  (GPT 5.4) 04/12/25 - 22/03/26
#DECLARATION: CODE HERE IS ASSISTED BY Copilot (GPT 5.4) 22/03/25 - 23/04/26

#region Imports
from importlib import import_module

import dash
from dash import dcc, html
try:
    from dash import Input, Output, State, callback_context
except ImportError:
    dash_dependencies = import_module('dash.dependencies')
    Input = dash_dependencies.Input
    Output = dash_dependencies.Output
    State = dash_dependencies.State
    callback_context = dash.callback_context
import numpy as np
#endregion

#region layouts for test and home pages

#this is the original layout dashboard - probably wont be used anymore
def level0_layout():
    return html.Div(
    #html.H1("Level 1 - Hyperparameter Playground")
    style={
        "height": "100vh",
        "padding": "10px",
        "backgroundColor": "#f3f4f6",
        "fontFamily": "Arial, sans-serif",
        "boxSizing": "border-box",
    },
    children=[
        html.H1(
            "Interactive Neural Network Dashboard",
            style={
                "textAlign": "center",
                "marginBottom": "10px",
                "fontSize": "28px",
            },
        ),
        #2x2 grid layout as discussed in new layout plan
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1.5fr",
                "gridTemplateRows": "1fr 1.2fr",
                "gap": "10px",
                "height": "calc(100% - 60px)",  #fill viewport minus title
            },
            children=[ #the four containers are the children
                #=========
                #2.1 Top left - Information box, 3 top buttons "Introduction", "Theory", "Tasks"
                #=========
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
                        "padding": "12px 16px",
                        "display": "flex",
                        "flexDirection": "column",
                        "minHeight": 0,
                    },
                    children=[
                        html.Div(
                            [
                                html.Span(
                                    "Information", #title for information box
                                    style={
                                        "fontSize": "18px",
                                        "fontWeight": "600",
                                        "marginRight": "12px",
                                    },
                                ),
                                html.Button( #this will button open a page for users to understand the dashboard purpose
                                    "Introduction",
                                    id="info-intro-btn",
                                    n_clicks=0,
                                    style={"marginRight": "6px", "padding": "4px 8px"},
                                ),
                                html.Button( #this button will provide mathematical concepts and background reading for FNN
                                    "Theory",
                                    id="info-theory-btn",
                                    n_clicks=0,
                                    style={"marginRight": "6px", "padding": "4px 8px"},
                                ),
                                html.Button( #walkthrough tasks to guide the user to a higher conceptual understanding of FNN
                                    "Tasks",
                                    id="info-tasks-btn",
                                    n_clicks=0,
                                    style={"padding": "4px 8px"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "marginBottom": "8px",
                            },
                        ),
                        html.Hr(style={"margin": "6px 0 10px 0"}),
                        html.Div(
                            id="info-content",
                            style={
                                "flex": "1",
                                "overflowY": "auto",
                                "fontSize": "13px",
                                "lineHeight": "1.4",
                                "whiteSpace": "pre-line",
                            },
                        ),
                    ],
                ),

                #=========
                #2.2 Top right - Feed forward neural network architecture
                #=========
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
                        "padding": "12px",
                        "display": "flex",
                        "flexDirection": "column",
                    },
                    children=[
                        html.Div(
                            "Feed-Forward Neural Network Architecture",
                            style={
                                "fontSize": "16px",
                                "fontWeight": "600",
                                "marginBottom": "6px",
                            },
                        ),
                        html.Div(
                            f"Train: 120 | Val: 30 | "
                            f"Test: 30 | Features: 4 | Classes: 3",
                            style={"fontSize": "11px", "color": "#4b5563", "marginBottom": "4px"},
                        ),
                        dcc.Graph(
                            id="architecture-graph",
                            style={
                                "flex": "1",
                                "minHeight": "0",
                            },
                            config={"displayModeBar": False},
                        ),
                    ],
                ),

                #=========
                #2.3 Bottom left - Hyperparameter Control panel (FNN architecture config and Training setup config)
                #=========
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
                        "padding": "12px 16px",
                        "display": "flex",
                        "flexDirection": "column",
                        "minHeight": 0,
                    },
                    children=[
                        html.Div(
                            [
                                html.Span(
                                    "Control Panel",
                                    style={
                                        "fontSize": "16px",
                                        "fontWeight": "600",
                                        "marginRight": "10px",
                                    },
                                ),
                                html.Button(
                                    "Train",
                                    id="train-btn",
                                    n_clicks=0,
                                    style={"marginRight": "6px", "padding": "6px 12px"},
                                ),
                                html.Button(
                                    "Reset",
                                    id="reset-btn",
                                    n_clicks=0,
                                    style={"padding": "6px 12px"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "marginBottom": "6px",
                            },
                        ),
                        html.Div(
                            id="status-text",
                            style={"fontSize": "12px", "color": "green", "marginBottom": "6px"},
                        ),
                        html.Hr(style={"margin": "4px 0 8px 0"}),
                        # Scrollable hyperparameters area
                        html.Div(
                            style={
                                "flex": "1",
                                "overflowY": "auto",
                                "paddingRight": "4px",
                                "fontSize": "13px",
                            },
                            children=[
                                html.H4("FNN Model Selection"),
                                dcc.Dropdown(["Logistic Regression", "NN-1-Layer", "NN-2-Layer"],
                                             "NN-1-Layer",
                                             id="model_dropdown"
                                             ),

                                html.H4("Dataset Selection", style={"fontSize": "14px", "marginBottom": "8px"}),
                                dcc.Dropdown(["Iris (Flowers)",
                                             "Wine (Chemistry)",
                                             "Seeds"],
                                             "Iris (Flowers)", #added s, might fix the lack of text initally
                                             id="ds_dropdown"
                                             ),
                                
                                html.H4(
                                    "Feed Forward Model Hyperparameters",
                                    style={"fontSize": "14px", "marginBottom": "8px"},
                                ),

                                html.Label("Weight Initialisation Seed:", style={"fontSize": "12px"}),
                                dcc.Input(
                                    id="seed-input",
                                    type="number",
                                    value=42,
                                    style={
                                        "width": "100%",
                                        "padding": "6px",
                                        "marginBottom": "12px",
                                    },
                                ),
                                html.Label("Hidden Layers Count:", style={"fontSize": "12px"}),
                                dcc.Slider(
                                    id="hidden-layer-count",
                                    min=1,
                                    max=15,
                                    step=1,
                                    value=8,
                                    marks={1: "1", 5: "5", 10: "10", 15: "15"},
                                ),

                                html.Label("Hidden Layer Size:", style={"fontSize": "12px"}),
                                dcc.Slider(
                                    id="hidden-size",
                                    min=1,
                                    max=15,
                                    step=1,
                                    value=5,
                                    marks={1: "1", 5: "5", 10: "10", 15: "15"},
                                ),

                                html.Label("Activation Functions:", style={"fontSize": "12px"}),
                                dcc.Dropdown(['Sigmoid', 
                                              'Tanh', 
                                              'Relu'], 
                                              'Sigmoid', 
                                              id='act-dropdown'),

                                html.Div(style={"height": "16px"}),

                                #separate section for training setup inputs
                                html.H4(
                                    "Training Setup Hyperparameters",
                                    style={"fontSize": "14px", "marginBottom": "4px"},
                                ),
                                html.Label("Gradient Descent Optimiser Algorithm:", style={"fontSize": "12px"}),
                                dcc.Dropdown(['Batch', 
                                              'Mini-Batch', 
                                              'Stochastic'], 
                                              'Batch', 
                                              id='optimiser-dropdown'),

                                html.Label("Learning Rate (log10):", style={"fontSize": "12px"}),
                                dcc.Slider(
                                    id="learning-rate",
                                    min=-3,
                                    max=-1,
                                    step=0.1,
                                    value=-2,
                                    marks={-3: "0.001", -2: "0.01", -1: "0.1"},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),

                                html.Label("Epochs:", style={"fontSize": "12px"}),
                                dcc.Slider(
                                    id="epochs",
                                    min=10,
                                    max=100,
                                    step=10,
                                    value=50,
                                    marks={10: "10", 50: "50", 100: "100"},
                                ),

                                html.Label("Batch-size:", style={"fontSize": "12px"}),
                                dcc.Slider(
                                    id="batch-size",
                                    min=10,
                                    max=100,
                                    step=10,
                                    value=50,
                                    marks={10: "10", 50: "50", 100: "100"},
                                ),

                                html.Label("Regularisation Strength (λ) :", style={"fontSize": "12px"}),
                                dcc.Slider(
                                    id="reg-strength",
                                    min=10,
                                    max=100,
                                    step=10,
                                    value=50,
                                    marks={10: "10", 50: "50", 100: "100"},
                                ),

                                html.Label("Early Stopping Criteria:", style={"fontSize": "12px"}),
                                dcc.Dropdown(['Validation Loss Plateu (Generalisation)', 
                                              'Validation Accuracy Plateu (Generalisation)'], 
                                              'Validation Loss Plateu (Generalisation)', 
                                              id='es-dropdown'),
                            ],
                        ),
                    ],
                ),



                #=========
                #2.4 Bottom right - Training outcomes (loss graphs, confusion matrices, accuracy metrics etc)
                #=========
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "8px",
                        "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
                        "padding": "12px",
                        "display": "flex",
                        "overflow": "hidden",
                        "flexDirection": "column",
                        "minHeight": 0,
                        "height": "100%",
                    },
                    children=[
                        html.Div(
                            "Training Outcomes",
                            style={
                                "fontSize": "16px",
                                "fontWeight": "600",
                                "marginBottom": "6px",
                            },
                        ),
                        # Training curves validation and accuracy
                        html.Div(
                            style={
                                "height": "50%",
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "8px",
                                "overflowY": "auto",
                                "overflowX": "hidden",
                            },
                            children=[
                                #loss curves (validation)
                                dcc.Graph(
                                    id="loss-curves",
                                    style={"flex": "1"},
                                    config={"displayModeBar": False},
                                ),

                                #accuracy curves (training)
                                dcc.Graph(
                                    id="accuracy-curves",
                                    style={"flex": "1"},
                                    config={"displayModeBar": False},
                                ),

                            ],
                        ),


                        html.Div(
                            style={
                                "display": "flex",
                                "marginTop": "8px",
                                "height": "60%",
                                "gap": "8px",
                            },
                            children=[
                                #confusion matrices
                                html.Div(
                                    style={"flex": "1", "minWidth": "0"},
                                    children=[
                                        dcc.Graph(
                                            id="confusion-matrix-heatmap",
                                            style={"height": "100%"},
                                            config={"displayModeBar": False},
                                        )
                                    ],
                                ),
                                # Metrics (can scroll them)
                                html.Div(
                                    style={
                                        "flex": "1",
                                        "minWidth": "0",
                                        "display": "flex",
                                        "flexDirection": "column",
                                    },
                                    children=[
                                        html.Div(
                                            id="accuracy-metrics",
                                            style={
                                                "fontSize": "12px",
                                                "marginBottom": "6px",
                                            },
                                        ),
                                        html.Div(
                                            id="per-class-metrics",
                                            style={
                                                "flex": "1",
                                                "overflowY": "auto",
                                                "fontSize": "11px",
                                            },
                                        ),

                                    ],
                                ),


                            ],
                        ),

                    ],

                ),

            ],

        ),
        dcc.Store(id="model-history-store"),
    ],
)

def home_layout(): #removed the type error by splitting it out
    return html.Div([
            html.H2("HOME PAGE: WELCOME TO NEURAL NETWORKS",
                    style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            html.H2("Welcome to Neural Network Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            html.H2("Skill tree to access levels, sandbox to see python coding enviroment.", 
                   style={'textAlign': 'center', 'fontSize': '18px'}),
                   #INTRODUCTION_TEXT  # Reuse existing intro content (maybe)
            
    ])
#endregion

#region SECURE PROGRAMMING: layout for sandbox code
def sandbox_layout():
    return html.Div([
        html.H2("SANDBOX",
                style={'textAlign': 'center', 'marginBottom': '20px'}),
        #instructions for ML pipeline
        html.Div([
            html.H3("Build your own ML pipeline from scratch."),
            html.H2("1. Load data"),
            html.H2("2. Feature engineering"),
            html.H2("3. Train Model"),
            html.H2("4. Evaluate"),
        ]),
        
        html.Div([
            html.Label("Python Code Editor:", style={"fontWeight": "bold"}),
            
            #code editor container
            html.Div(
                dcc.Textarea(
                    id="code-input",
                    value="print('Hello World')",
                    style={
                        "width": "100%",
                        "height": "400px",
                        "fontFamily": "Consolas, Monaco, monospace",
                        "fontSize": "14px", 
                        "lineHeight": "1.4"
                        }
                ),
            ),

            #controls html frontend

            html.Div([
                html.Button("Run Code", id="code-run", #to run the python code
                       style={"width": "100%", "padding": "12px", 
                              "fontSize": "16px", "marginTop": "10px"}),
                
                html.Button("Export Code", id="code-export", #to export the python code
                        style={"width": "100%", "padding": "12px", 
                              "fontSize": "16px", "marginTop": "10px"}),

                dcc.Download(id="download-editor")
            ], style={"marginBottom": "20px"}),
           
           #output html panels

           html.Div(id="editor-output"),
           html.Div(id="editor-error", style={"color": "#dc3545"}),
           dcc.Graph(id="editor-plot")], style={"maxWidth": "1400px", "margin": "0 auto"})


    ])
#endregion

#region ADAPTIVE LEARNING: hopefully this can be be integrated with gamification features
#skill tree data as a placeholder to showcase clicing on different levels at different points
SKILL_TREE_DATA = {
    "nodes": [
        {"id": "level1", "name": "Hyperparams", "x": 0, "y": 0, "unlocked": True, "completed": False},
        {"id": "level2", "name": "Templates", "x": 1, "y": 0, "unlocked": False, "completed": False},
        {"id": "level3", "name": "Functions", "x": 2, "y": 0, "unlocked": False, "completed": False},
        {"id": "level4", "name": "Classes", "x": 1, "y": 1, "unlocked": False, "completed": False},
        {"id": "level5", "name": "Optimizers", "x": 2, "y": 1, "unlocked": False, "completed": False},
    ],
    "prereqs": {
        "level2": ["level1"],
        "level3": ["level2"], 
        "level4": ["level2"],
        "level5": ["level3", "level4"]
    }
}
def skilltree_layout():
    return html.Div([
        html.H1("SKILL TREE",
                    style={'textAlign': 'center', 'marginBottom': '20px'}),

        html.Div([
            dcc.Link([skill_box("Level 5", "Hidden Representations & Feature Spaces")], href="/level5"),      # row 1

            dcc.Link([skill_box("Level 4", "Training via Loss & Backprop")], href="/level4"),         # row 2

            dcc.Link([skill_box("Level 3", "Deeper Networks & Expressivity")], href="/level3"),       # row 3

            dcc.Link([skill_box("Level 2", "Architecture Impact")], href="/level2"),       # row 4

            dcc.Link([skill_box("Level 1", "Preconfigured FFNN Explorer")], href="/level1"), # row 5
        ])            
    ], style={
            "display": "grid",
            "gridTemplateColumns": "repeat(1, 1fr)",
            "gap": "20px",
            "maxWidth": "900px",
            "margin": "0 auto",
            "padding": "20px"
    })
def skill_box(title, subtitle):
    return html.Div([
        html.H3(title),
        html.P(subtitle)
    ], className = "skill-box")
#endregion

#region Level layouts

#i think here i should reduce the levels down to features rather than concepts

##1. Level 1 - hyperparameter sliders with PRECONFIGURED model to show instant visual feedback
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

#level 2 - model builder with UI
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

#3. level 3 - building a model with multiple code cells (visualisation of intermediate outputs)
def level3_layout():
    return html.Div([
        html.H2(
            "Level 3 – Deeper Networks and Expressivity",
            style={'textAlign': 'center', 'marginBottom': '10px'}
        ),

        html.P(
            "Stacking layers applies repeated space transformations. "
            "More neurons and layers let the network approximate richer functions "
            "(universal approximation). Use the sliders to watch piecewise-linear "
            "approximations become more accurate.",
            style={'textAlign': 'center', 'marginBottom': '20px', 'maxWidth': '800px',
                   'margin': '0 auto 20px auto'}
        ),

        # Controls row
        html.Div([
            html.Div([
                html.H4("Target function"),
                dcc.Dropdown(
                    id='level3-target-dropdown',
                    options=[
                        {'label': 'sin(x)', 'value': 'sin'},
                        {'label': 'abs(x)', 'value': 'abs'},
                        {'label': 'x² (quadratic)', 'value': 'quadratic'},
                        {'label': 'step function', 'value': 'step'},
                        {'label': 'sawtooth', 'value': 'sawtooth'},
                    ],
                    value='sin',
                    clearable=False
                ),

                html.H4("Hidden width (neurons per layer)"),
                dcc.Slider(
                    id='level3-width-slider',
                    min=1, max=32, step=1, value=8,
                    marks={1: '1', 4: '4', 8: '8', 16: '16', 32: '32'}
                ),

                html.H4("Depth (number of hidden layers)"),
                dcc.Slider(
                    id='level3-depth-slider',
                    min=1, max=6, step=1, value=1,
                    marks={i: str(i) for i in range(1, 7)}
                ),

                html.H4("Activation function"),
                dcc.Dropdown(
                    id='level3-activation-dropdown',
                    options=[
                        {'label': 'ReLU', 'value': 'relu'},
                        {'label': 'Tanh', 'value': 'tanh'},
                        {'label': 'Sigmoid', 'value': 'sigmoid'},
                    ],
                    value='relu',
                    clearable=False
                ),

                html.H4("Training epochs per click"),
                dcc.Slider(
                    id='level3-epochs-slider',
                    min=50, max=500, step=50, value=200,
                    marks={50: '50', 200: '200', 500: '500'}
                ),

                html.Div([
                    html.Button('Randomize & Reset', id='level3-randomize-btn', n_clicks=0,
                                style={'marginRight': '8px'}),
                    html.Button('Train', id='level3-train-btn', n_clicks=0),
                ], style={'marginTop': '12px'}),

            ], style={'width': '28%', 'display': 'inline-block',
                      'verticalAlign': 'top', 'padding': '0 20px'}),

            # Side-by-side: target vs approximation
            html.Div([
                dcc.Graph(id='level3-approx-graph', style={'height': '50vh'}),
            ], style={'width': '70%', 'display': 'inline-block'}),
        ]),

        html.Hr(),

        # Bottom row: network info + parameter count + loss curve
        html.Div([
            html.Div([
                html.H4("Network architecture summary"),
                html.Div(id='level3-arch-summary'),
            ], style={'width': '33%', 'display': 'inline-block',
                      'verticalAlign': 'top', 'padding': '0 20px'}),

            html.Div([
                html.H4("Training loss curve"),
                dcc.Graph(id='level3-loss-graph'),
            ], style={'width': '33%', 'display': 'inline-block'}),

            html.Div([
                html.H4("Per-layer activations"),
                dcc.Graph(id='level3-activations-graph'),
            ], style={'width': '33%', 'display': 'inline-block'}),
        ]),

        # Store for model parameters and training history
        dcc.Store(id='level3-params-store'),
    ])

def level4_layout():
    return html.Div([
        html.H2(
            "Level 4 – Training via Loss and Backpropagation",
            style={'textAlign': 'center', 'marginBottom': '10px'}
        ),

        html.P(
            "Explore how networks learn: observe the loss surface, watch gradient "
            "descent update weights, step through forward and backward passes, "
            "and develop intuition for learning rate and over/under-fitting.",
            style={'textAlign': 'center', 'marginBottom': '20px', 'maxWidth': '850px',
                   'margin': '0 auto 20px auto'}
        ),

        # Top row: controls + decision boundary animation
        html.Div([
            # Left: controls
            html.Div([
                html.H4("Toy dataset"),
                dcc.Dropdown(
                    id='level4-dataset-dropdown',
                    options=[
                        {'label': 'Moons', 'value': 'moons'},
                        {'label': 'Circles', 'value': 'circles'},
                        {'label': 'Linear', 'value': 'linear'},
                    ],
                    value='moons', clearable=False
                ),

                html.H4("Hidden width"),
                dcc.Slider(
                    id='level4-width-slider',
                    min=2, max=16, step=1, value=6,
                    marks={2: '2', 4: '4', 8: '8', 12: '12', 16: '16'}
                ),

                html.H4("Activation function"),
                dcc.Dropdown(
                    id='level4-activation-dropdown',
                    options=[
                        {'label': 'ReLU', 'value': 'relu'},
                        {'label': 'Tanh', 'value': 'tanh'},
                        {'label': 'Sigmoid', 'value': 'sigmoid'},
                    ],
                    value='tanh', clearable=False
                ),

                html.H4("Learning rate"),
                dcc.Slider(
                    id='level4-lr-slider',
                    min=-3, max=0, step=0.1, value=-1.5,
                    marks={-3: '0.001', -2: '0.01', -1: '0.1', 0: '1.0'},
                    tooltip={'placement': 'bottom', 'always_visible': True}
                ),

                html.H4("Train / validation split"),
                dcc.Slider(
                    id='level4-split-slider',
                    min=0.1, max=0.5, step=0.05, value=0.2,
                    marks={0.1: '10%', 0.2: '20%', 0.3: '30%', 0.5: '50%'}
                ),

                html.Div([
                    html.Button('Reset & Randomize', id='level4-reset-btn', n_clicks=0,
                                style={'marginRight': '8px'}),
                    html.Button('Train 1 Epoch', id='level4-step-btn', n_clicks=0,
                                style={'marginRight': '8px'}),
                    html.Button('Train 50 Epochs', id='level4-train50-btn', n_clicks=0),
                ], style={'marginTop': '12px'}),

                html.Div(id='level4-epoch-counter',
                         style={'marginTop': '8px', 'fontWeight': 'bold'}),

            ], style={'width': '28%', 'display': 'inline-block',
                      'verticalAlign': 'top', 'padding': '0 20px'}),

            # Right: decision boundary
            html.Div([
                dcc.Graph(id='level4-boundary-graph', style={'height': '50vh'}),
            ], style={'width': '70%', 'display': 'inline-block'}),
        ]),

        html.Hr(),

        # Middle row: loss curves + gradient flow diagram
        html.Div([
            html.Div([
                html.H4("Training & validation loss curves"),
                dcc.Graph(id='level4-loss-graph'),
            ], style={'width': '33%', 'display': 'inline-block'}),

            html.Div([
                html.H4("Training & validation accuracy"),
                dcc.Graph(id='level4-accuracy-graph'),
            ], style={'width': '33%', 'display': 'inline-block'}),

            html.Div([
                html.H4("Backprop gradient flow"),
                dcc.Graph(id='level4-gradient-graph'),
            ], style={'width': '33%', 'display': 'inline-block'}),
        ]),

        html.Hr(),

        # Bottom row: step-through explanation
        html.Div([
            html.Div([
                html.H4("Forward / Backward pass walkthrough"),
                html.Div(id='level4-pass-explanation'),
            ], style={'width': '50%', 'display': 'inline-block',
                      'verticalAlign': 'top', 'padding': '0 20px'}),

            html.Div([
                html.H4("Fitting status & insights"),
                html.Div(id='level4-fit-status'),
            ], style={'width': '50%', 'display': 'inline-block',
                      'verticalAlign': 'top', 'padding': '0 20px'}),
        ]),

        # Stores
        dcc.Store(id='level4-params-store'),
    ])

def level5_layout():
    return html.Div([
        html.H2(
            "Level 5 \u2013 Hidden Representations and Feature Spaces",
            style={'textAlign': 'center', 'marginBottom': '10px'}
        ),

        html.P(
            "Watch what hidden layers actually do: each layer re-maps the data into a new "
            "feature space. Slide through layers to see the input data morph until the "
            "final hidden layer makes the classes linearly separable.",
            style={'textAlign': 'center', 'marginBottom': '20px', 'maxWidth': '850px',
                   'margin': '0 auto 20px auto'}
        ),

        # ── top row: controls + feature-space scatter ──
        html.Div([
            # Left column: controls
            html.Div([
                html.H4("Toy dataset"),
                dcc.Dropdown(
                    id='level5-dataset-dropdown',
                    options=[
                        {'label': 'Moons', 'value': 'moons'},
                        {'label': 'Circles', 'value': 'circles'},
                        {'label': 'Linear', 'value': 'linear'},
                    ],
                    value='moons', clearable=False
                ),

                html.H4("Network depth (hidden layers)"),
                dcc.Slider(
                    id='level5-depth-slider',
                    min=1, max=5, step=1, value=3,
                    marks={i: str(i) for i in range(1, 6)}
                ),

                html.H4("Hidden width"),
                dcc.Slider(
                    id='level5-width-slider',
                    min=2, max=16, step=1, value=6,
                    marks={2: '2', 4: '4', 8: '8', 12: '12', 16: '16'}
                ),

                html.H4("Activation function"),
                dcc.Dropdown(
                    id='level5-activation-dropdown',
                    options=[
                        {'label': 'ReLU', 'value': 'relu'},
                        {'label': 'Tanh', 'value': 'tanh'},
                        {'label': 'Sigmoid', 'value': 'sigmoid'},
                    ],
                    value='tanh', clearable=False
                ),

                html.Div([
                    html.Button('Reset & Randomize', id='level5-reset-btn', n_clicks=0,
                                style={'marginRight': '8px'}),
                    html.Button('Train 50 Epochs', id='level5-train50-btn', n_clicks=0,
                                style={'marginRight': '8px'}),
                    html.Button('Train 200 Epochs', id='level5-train200-btn', n_clicks=0),
                ], style={'marginTop': '12px'}),

                html.Div(id='level5-epoch-counter',
                         style={'marginTop': '8px', 'fontWeight': 'bold'}),

                html.Hr(),
                html.H4("Layer slider"),
                html.P("Slide to see the data after each layer:",
                       style={'fontSize': '12px'}),
                dcc.Slider(
                    id='level5-layer-slider',
                    min=0, max=3, step=1, value=0,
                    marks={0: 'Input'},
                    tooltip={'placement': 'bottom', 'always_visible': True}
                ),

            ], style={'width': '28%', 'display': 'inline-block',
                      'verticalAlign': 'top', 'padding': '0 20px'}),

            # Right column: feature-space scatter
            html.Div([
                html.H4("Feature-space view (layer embedding)",
                        style={'textAlign': 'center'}),
                dcc.Graph(id='level5-feature-graph', style={'height': '55vh'}),
            ], style={'width': '70%', 'display': 'inline-block'}),
        ]),

        html.Hr(),

        # ── middle row: decision boundary + loss curve ──
        html.Div([
            html.Div([
                html.H4("Decision boundary (input space)"),
                dcc.Graph(id='level5-boundary-graph'),
            ], style={'width': '50%', 'display': 'inline-block'}),

            html.Div([
                html.H4("Training loss curve"),
                dcc.Graph(id='level5-loss-graph'),
            ], style={'width': '50%', 'display': 'inline-block'}),
        ]),

        html.Hr(),

        # ── bottom row: interpretability explanation ──
        html.Div([
            html.Div([
                html.H4("What is happening at this layer?"),
                html.Div(id='level5-layer-explanation'),
            ], style={'width': '50%', 'display': 'inline-block',
                      'verticalAlign': 'top', 'padding': '0 20px'}),

            html.Div([
                html.H4("Linear separability check"),
                html.Div(id='level5-separability-info'),
            ], style={'width': '50%', 'display': 'inline-block',
                      'verticalAlign': 'top', 'padding': '0 20px'}),
        ]),

        # Stores
        dcc.Store(id='level5-params-store'),
    ])

#endregion