import dash
from dash import dcc, html, Input, Output, State, callback_context
import numpy as np
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

def student_layout():
    return html.Div(
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
                                             "Iris (Flower)",
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
            
            html.H2("Use the navigation above to access Student or Teacher views.", 
                   style={'textAlign': 'center', 'fontSize': '18px'}),
                   #INTRODUCTION_TEXT  # Reuse existing intro content
            
    ])