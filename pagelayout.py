import dash
from dash import dcc, html, Input, Output, State, callback_context
import numpy as np

"""
def teacher_layout():
    return html.Div([
        html.H2("Teacher Page - Lesson setup"),
        html.Label("Dataset"),
        dcc.Dropdown(
            ["Iris", "Wine", "Seeds"],
            "Iris",
            id="teacher-dataset-dropdown",
        ),

        html.Label("Model Type"),
        dcc.Dropdown(
            ["Logistic Regression", "NN-1-Layer", "NN-2-Layer"],
            "NN-1-Layer",
            id="teacher-model-type",
        ),

        html.Label("Hidden size (for NN)"),
        dcc.Slider(1, 32, 1, value=8, id="teacher-hidden-size"),

        html.Label("Learning rate (log10)"),
        dcc.Slider(-3, -1, 0.1, value=2, id="teacher-learning-rate"),

        html.Label("Epochs"),
        dcc.Slider(0, 50, 100, value=50, id="teacher-epochs"),

        html.Label("Student Control Permissive"),
        dcc.Checklist(
            options=[
                {"label": "Hidden size", "value": "hidden_size"},
                {"label": "Learning rate", "value": "lr"},
                {"label": "Epochs", "value": "epochs"},
            ],
            value=["hidden_size", "lr"],
            id="teacher-controls",
        ),

        #area to write and save a lesson for the student

        html.Label("Lesson instructions"),
        dcc.Textarea(
            id="tasks-text",
            style={"width": "100%", "height": 120},
            placeholder="1) Look at overfitting",
        ),

        html.Button("Save lesson preset",
                    id="save-lesson-btn",
                    n_clicks=0),
        html.Div(id="teacher-status",
                 style={"marginTop": "10px,", "color": "green"}),

    ]),
"""
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

#new levels

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
            dcc.Link([skill_box("Level 5", "The ML Pipeline")], href="/level5"),      # row 1

            dcc.Link([skill_box("Level 4", "Model Architecture Coding")], href="/level4"),         # row 2

            dcc.Link([skill_box("Level 3", "Deeper Networks & Expressivity")], href="/level3"),       # row 3

            dcc.Link([skill_box("Level 2", "Architecture Impact")], href="/level2"),       # row 4

            dcc.Link([skill_box("Level 1", "Linear Decision Boundary Explorer")], href="/level1"), # row 5
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

"""
def level1_layout():
    return html.Div([
        html.H1("Level 1: Hyperparameter Playground", style={'textAlign': 'center'}),
        
        # Right side of screen  = Controls + Code Preview
        html.Div([
            html.H3("Change Architecture Live"),
            # sliders (Req 1.1.3)
            html.Label("Hidden Layer Size:"), dcc.Slider(id='live-hidden-size', min=2, max=16, value=8,
                                                        marks={i: str(i) for i in [2,4,8,12,16]}),
            html.Label("Random Seed:"), dcc.Input(id='live-seed', type='number', value=42),
            
            # code preview (Req 3.1.6)
            html.Label("Live Code Preview:"), 
            html.Div(id='code-preview', children=[
                dcc.Textarea(value="model = SimpleNN(4, 8, 3, seed=42)", 
                           style={'width': '100%', 'height': 100, 'fontFamily': 'monospace'})
            ]),
            html.Button("Test Architecture", id='test-arch-btn')
        ], style={'width': '45%', 'float': 'left'}),
        
        # right side of screen: this will be live visualisation from ui sliders (Req 3.1.5)
        html.Div([
            dcc.Graph(id='live-architecture'),      # Nodes + edges
            dcc.Graph(id='live-weights-heatmap'),   # Parameter matrices
            dcc.Graph(id='live-decision-boundary')  # Classification viz
        ], style={'width': '55%', 'float': 'right'})
    ])
"""
def level1_layout():
    return html.Div(
        [
            html.H1(
                "Level 1: Linear Decision Boundary Explorer",
                style={'textAlign': 'center'}
            ),

            # Parent flex container
            html.Div(
                [
                    # Left side: controls
                    html.Div(
                        [
                            html.H3("Data & Linear Model Controls"),

                            html.Label("Dataset:"),
                            dcc.Dropdown(
                                id='l1-dataset',
                                options=[
                                    {'label': 'Linearly Separable', 'value': 'linear'},
                                    {'label': 'Not Linearly Separable (Moons)', 'value': 'moons'},
                                    {'label': 'Not Linearly Separable (Circles)', 'value': 'circles'}
                                ],
                                value='linear',
                                clearable=False
                            ),

                            html.Hr(),

                            html.Label("Weight w₁ (x-axis):"),
                            dcc.Slider(
                                id='l1-w1',
                                min=-5, max=5, step=0.1, value=1.0,
                                marks={i: str(i) for i in range(-5, 6, 2)}
                            ),

                            html.Label("Weight w₂ (y-axis):"),
                            dcc.Slider(
                                id='l1-w2',
                                min=-5, max=5, step=0.1, value=1.0,
                                marks={i: str(i) for i in range(-5, 6, 2)}
                            ),

                            html.Label("Bias b:"),
                            dcc.Slider(
                                id='l1-bias',
                                min=-5, max=5, step=0.1, value=0.0,
                                marks={i: str(i) for i in range(-5, 6, 2)}
                            ),

                            html.Hr(),

                            html.Label("Learning Rate (for training step):"),
                            dcc.Slider(
                                id='l1-lr',
                                min=0.001, max=1.0, step=0.001, value=0.1,
                                marks={0.001: '0.001', 0.01: '0.01', 0.1: '0.1', 1.0: '1.0'}
                            ),

                            html.Div(
                                [
                                    html.Button("Train One Step", id='l1-train-step', n_clicks=0),
                                    html.Button(
                                        "Reset Weights",
                                        id='l1-reset',
                                        n_clicks=0,
                                        style={'marginLeft': '10px'}
                                    )
                                ],
                                style={'marginTop': '10px'}
                            ),

                            html.Hr(),

                            html.Label("Current Linear Model:"),
                            dcc.Textarea(
                                id='l1-model-equation',
                                value="sign(w1 * x + w2 * y + b)",
                                style={
                                    'width': '100%',
                                    'height': 60,
                                    'fontFamily': 'monospace'
                                }
                            ),
                        ],
                        style={
                            'flex': '0 0 35%',
                            'padding': '0 20px',
                            'boxSizing': 'border-box'
                        }
                    ),

                    # Right side: visualisations
                    html.Div(
                        [
                            html.H3("Decision Boundary & Loss"),

                            dcc.Graph(
                                id='l1-decision-boundary',
                                style={'height': '55vh'}
                            ),

                            dcc.Graph(
                                id='l1-loss-curve',
                                style={'height': '30vh', 'marginTop': '10px'}
                            ),
                        ],
                        style={
                            'flex': '0 0 65%',
                            'padding': '0 20px',
                            'boxSizing': 'border-box'
                        }
                    ),
                ],
                style={
                    'display': 'flex',
                    'flexDirection': 'row',
                    'alignItems': 'flex-start'
                }
            ),
        ]
    )


def level2_layout():
    return html.Div([
        html.H2(
            "Level 2 – Single Hidden Layer as Function Composition",
            style={'textAlign': 'center', 'marginBottom': '20px'}
        ),

        html.P(
            "Explore how a single hidden layer applies a linear transformation "
            "followed by a nonlinearity to bend the decision boundary.",
            style={'textAlign': 'center', 'marginBottom': '20px'}
        ),

        # Controls row
        html.Div([
            html.Div([
                html.H4("Hidden layer width"),
                dcc.Slider(
                    id='level2-width-slider',
                    min=1,
                    max=10,
                    step=1,
                    value=4,
                    marks={i: str(i) for i in range(1, 11)}
                ),

                html.H4("Activation function"),
                dcc.Dropdown(
                    id='level2-activation-dropdown',
                    options=[
                        {'label': 'ReLU', 'value': 'relu'},
                        {'label': 'Tanh', 'value': 'tanh'},
                        {'label': 'Sigmoid', 'value': 'sigmoid'},
                    ],
                    value='tanh',
                    clearable=False
                ),

                html.H4("Toy dataset"),
                dcc.Dropdown(
                    id='level2-dataset-dropdown',
                    options=[
                        {'label': 'Moons', 'value': 'moons'},
                        {'label': 'Circles', 'value': 'circles'},
                        {'label': 'Linear', 'value': 'linear'},
                    ],
                    value='moons',
                    clearable=False
                ),

                html.Button('Randomize Weights', id='level2-randomize-btn', n_clicks=0),
                html.Button('Train Few Steps', id='level2-trainstep-btn', n_clicks=0),
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '0 20px'}),

            html.Div([
                dcc.Graph(id='level2-decision-boundary-graph'),
            ], style={'width': '70%', 'display': 'inline-block'}),
        ]),

        html.Hr(),

        # Second row: activation + network diagram + math explanation
        html.Div([
            html.Div([
                html.H4("Activation function ρ(z)"),
                dcc.Graph(id='level2-activation-graph'),
            ], style={'width': '33%', 'display': 'inline-block'}),

            html.Div([
                html.H4("Network diagram"),
                dcc.Graph(id='level2-network-diagram-graph'),
            ], style={'width': '33%', 'display': 'inline-block'}),

            html.Div([
                html.H4("Layer formula & parameters"),
                html.Div(id='level2-math-explanation'),
            ], style={'width': '33%', 'display': 'inline-block', 'padding': '0 20px'}),
        ]),

        # Store for model parameters (weights, biases)
        dcc.Store(id='level2-params-store'),
    ])

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
        html.H2("Full Architecture Coding",
                    style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        html.Div([
            html.H3("Build SimpleNN Class"),
            html.P("Your ReLU & Softmax from Level 3 must match models-4.py exactly"),
            html.Ul([
                html.Li("Copy SimpleNN signature below"),
                html.Li("Implement __init__, forward, backward, train_epoch"),
                html.Li("Match Level 2 training curves")
            ]),
            
            html.H4("Reference Template"),
            html.Pre("""class SimpleNN:"""),
        ], style={'width': '45%', 'float': 'left'}),

        html.Div([
            html.Label("Custom SimpleNN Implementation"),
            dcc.Textarea(id='level4-code-input', value="Level3 functions here",
            style = {'width': '100%', 'height': '400px'}),

            html.Button('Test Class', id='level4-test-btn', n_clicks=0),
            html.Button('Train & Compare', id='level4-train-btn', n_clicks=0),
            html.Div(id='level4-status')
            ], style={'width': '55%', 'float': 'right'}),

        html.Div(id='level4-output', style={'clear': 'both'}),
        html.Div([
            dcc.Graph(id='level4-loss-compare'),
            dcc.Graph(id='level4-weight-compare')
        ], style={'display': 'flex'})
            
    ])

def level5_layout():
    return html.Div([
        html.H1("LEVEL 5: Complete ML Pipeline", style={'textAlign': 'center'}),
        
        html.Div([
            html.H3("✅ Build Production Pipeline"),
            html.P("Integrate your SimpleNN + create optimizers"),
            html.Ul([
                html.Li("Load data: loaddataset('iris')"),
                html.Li("Custom SGD/Mini-batch optimizer"),
                html.Li("Early stopping + L2 regularization"),
                html.Li("Beat sklearn LogisticRegression (95%+ accuracy)")
            ])
        ], style={'width': '35%', 'float': 'left'}),
        
        # Full pipeline code editor
        html.Div([
            html.Label("Complete Pipeline"),
            dcc.Textarea(id='level5-code-input', value="""# YOUR SimpleNN FROM LEVEL 4 HERE
class SimpleNN:
    # Paste your working class

# Task 1: Custom Optimizer Class  
class CustomOptimizer:
    def __init__(self, lr=0.01, batch_size=32, momentum=0.9):
        self.lr = lr
        self.batch_size = batch_size  
        self.momentum = momentum
        self.v_w1 = self.v_b1 = self.v_w2 = self.v_b2 = 0
        
    def step(self, model, X_batch, y_batch):
        # Implement SGD + momentum updates
        pass

# Task 2: Training Pipeline
def train_pipeline(dataset='iris', epochs=100):
    # 1. Load data
    X_train, X_test, y_train, y_test, meta = loaddataset(dataset)
    
    # 2. Build model  
    model = SimpleNN(4, 8, 3, seed=42)
    optimizer = CustomOptimizer(lr=0.01, batch_size=32)
    
    # 3. Training loop with early stopping
    history = {'loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Your training code here
        pass
    
    return model, history

# RUN PIPELINE
model, history = train_pipeline()
print(f"Test Accuracy: {np.mean(np.argmax(model.forward(X_test), 1) == y_test):.3f}")
""", style={'width': '100%', 'height': '500px'}),
            
            html.Button('Run Pipeline', id='level5-run-btn', n_clicks=0),
            html.Button('Compare vs Sklearn', id='level5-compare-btn', n_clicks=0)
        ], style={'width': '65%', 'float': 'right'}),
        
        html.Div(id='level5-output', style={'clear': 'both', 'marginTop': '20px'}),
        
        # Full dashboard visuals
        html.Div([
            dcc.Graph(id='level5-loss-curves'),
            dcc.Graph(id='level5-confusion-matrix'),
            dcc.Graph(id='level5-accuracy-metrics')
        ], style={'display': 'flex', 'gap': '20px'})
    ])
