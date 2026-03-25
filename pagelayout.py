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
from code_execution import CodeExecutionBox

from levels.level1.layout import level1_layout


SANDBOX_EDITOR = CodeExecutionBox(
    'sandbox',
    ids={
        'input': 'code-input',
        'run': 'code-run',
        'export': 'code-export',
        'download': 'download-editor',
        'output': 'editor-output',
        'error': 'editor-error',
        'plot': 'editor-plot',
        'validation': 'code-validation',
        'highlighted': 'code-highlighted',
    },
)
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
            SANDBOX_EDITOR.render(
                default_code="print('Hello World')",
                title=None,
                run_label='Run Code',
                export_label='Export Code',
                show_export=True,
                include_plot=True,
                code_height='400px',
                wrapper_style={'padding': '0', 'boxShadow': 'none', 'backgroundColor': 'transparent'},
            )
        ], style={"maxWidth": "1400px", "margin": "0 auto"})


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
        {"id": "level5", "name": "Optimisers", "x": 2, "y": 1, "unlocked": False, "completed": False},
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

#level 2 - model builder with UI

def _level3_code_cell(title, description, code_value, button_id, button_text, controls=None):
    if controls is None:
        controls = []

    return html.Div([
        html.Div([
            html.Span(title, style={'fontWeight': '700', 'fontSize': '16px'}),
            html.Button(
                button_text,
                id=button_id,
                n_clicks=0,
                style={
                    'backgroundColor': '#0f766e',
                    'color': 'white',
                    'border': 'none',
                    'padding': '8px 12px',
                    'borderRadius': '8px'
                }
            )
        ], style={
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center',
            'marginBottom': '10px'
        }),
        html.P(description, style={'fontSize': '13px', 'color': '#475569', 'marginBottom': '12px'}),
        dcc.Textarea(
            value=code_value,
            readOnly=True,
            style={
                'width': '100%',
                'height': '140px',
                'fontFamily': 'Consolas, Monaco, monospace',
                'fontSize': '13px',
                'lineHeight': '1.5',
                'backgroundColor': '#0f172a',
                'color': '#e2e8f0',
                'border': 'none',
                'borderRadius': '10px',
                'padding': '12px',
                'boxSizing': 'border-box',
                'marginBottom': '12px'
            }
        ),
        html.Div(controls)
    ], style={
        'backgroundColor': 'white',
        'borderRadius': '16px',
        'padding': '18px',
        'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)',
        'marginBottom': '16px'
    })


def _level3_output_card(title, children):
    return html.Div([
        html.H3(title, style={'marginTop': '0', 'marginBottom': '14px'}),
        children,
    ], style={
        'backgroundColor': 'white',
        'borderRadius': '16px',
        'padding': '18px',
        'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)',
        'height': '100%',
        'boxSizing': 'border-box'
    })


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