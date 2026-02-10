import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
from dataload import load_dataset_iris, get_dataset_stats
from trainer import train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
 
# Load data once at startup
X_train_full, X_test, y_train_full, y_test, feature_names, class_names = load_dataset_iris()
#dataset_stats = get_dataset_stats(X_train, y_train)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)
 
app = dash.Dash(__name__)

# Disable caching for development
#app.contrain_fig.suppress_callback_exceptions = False
#app.contrain_fig.assets_folder = 'assets'

# Add cache busting headers
#@app.server.after_request
#def add_header(response):
#    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
#    response.headers['Pragma'] = 'no-cache'
#    response.headers['Expires'] = '0'
#    return response


#=======NEW PLAN=======


#1. Info box content variables = "Introduction", "Theory", "Tasks"

#2. Layout plan - 4 box grid dashboard
#2.1 Top left - Information box, 3 top buttons "Introduction", "Theory", "Tasks"
#2.2 Top right - Feed forward neural network architecture
#2.3 Bottom left - Hyperparameter Control panel (FNN architecture config and Training setup config)
#2.4 Bottom right - Training outcomes (loss graphs, confusion matrices, accuracy metrics etc)

#3. Callbacks
#3.1 Information box
#3.2 Hyperparameter Control 
#3.3 Reset trained model


#Information that could be used in the callback of the information box
INTRODUCTION_TEXT = html.Div([
    html.P("This dashboard provides an interactive walkthrough of a Feed-Forward Neural Network solving a classification problem with Iris dataset. \n"
    "It will provide a complementary experience of theory and tasks intended to improve conceptual understanding of Neural Network concepts."),
    html.Img(
        src="/assets/nn_diagram.png",
        style={
            "maxWidth": "100%",
            "height": "auto",
            "marginTop": "8px",
            "borderRadius": "4px"
        }
    ),
])

THEORY_TEXT = html.Div([
    html.P("Theory Text"),
    html.Img(
        src="/assets/conveyor_belt.png",
        style={
            "maxWidth": "100%",
            "height": "auto",
            "marginTop": "8px",
            "borderRadius": "4px"
        }
    ),
])

TASKS_TEXT = html.Div([
    html.P("Tasks Text"),
    html.Img(
        src="/assets/activation_functions.png",
        style={
            "maxWidth": "100%",
            "height": "auto",
            "marginTop": "8px",
            "borderRadius": "4px"
        }
    ),
])

#new layout - 4 box grid

app.layout = html.Div(
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
                            f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | "
                            f"Test: {X_test.shape[0]} | Features: {X_train.shape[1]} | Classes: 3",
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

                                html.Label("Input Layer Size:", style={"fontSize": "12px"}),
                                dcc.Slider(
                                    id="input-size",
                                    min=1,
                                    max=15,
                                    step=1,
                                    value=4,
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
                                html.Label("Output Layer Size:", style={"fontSize": "12px"}),
                                dcc.Slider(
                                    id="output-size",
                                    min=1,
                                    max=15,
                                    step=1,
                                    value=3,
                                    marks={1: "1", 5: "5", 10: "10", 15: "15"},
                                ),

                                html.Label("Activation Functions:", style={"fontSize": "12px"}),
                                dcc.Dropdown(['Sigmoid', 'Tanh', 'Relu'], 'Sigmoid', id='act-dropdown'),

                                html.Div(style={"height": "16px"}),

                                #separate section for training setup inputs
                                html.H4(
                                    "Training Setup Hyperparameters",
                                    style={"fontSize": "14px", "marginBottom": "4px"},
                                ),
                                html.Label("Gradient Descent Optimiser Algorithm:", style={"fontSize": "12px"}),
                                dcc.Dropdown(['Batch', 'Mini-Batch', 'Stochastic'], 'Batch', id='optimiser-dropdown'),

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
                                dcc.Dropdown(['Validation Loss Plateu (Generalisation)', 'Validation Accuracy Plateu (Generalisation)'], 'Validation Loss Plateu (Generalisation)', id='es-dropdown'),
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
                        "flexDirection": "column",
                        "minHeight": 0,
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
                        # Training curves
                        dcc.Graph(
                            id="training-curves",
                            style={"height": "40%"},
                            config={"displayModeBar": False},
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

#app.layout.children.append(dcc.Store(id="model-history-store"))


#3.1 information box callback

@app.callback(
    Output("info-content", "children"),
    Input("info-intro-btn", "n_clicks"),
    Input("info-theory-btn", "n_clicks"),
    Input("info-tasks-btn", "n_clicks"),
)
def update_info_content(n_intro, n_theory, n_tasks):
    # Determine which button was pressed most recently
    ctx = dash.callback_context
    if not ctx.triggered:
        # default view
        return INTRODUCTION_TEXT
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id == "info-theory-btn":
        return THEORY_TEXT
    elif button_id == "info-tasks-btn":
        return TASKS_TEXT
    else:
        return INTRODUCTION_TEXT
 
#callback for training outcomes
@app.callback(
    [
        Output('training-curves', 'figure'),
        Output('architecture-graph', 'figure'),
        Output('confusion-matrix-heatmap', 'figure'),
        Output('accuracy-metrics', 'children'),
        Output('per-class-metrics', 'children'),
        Output('status-text', 'children'),
        Output('model-history-store', 'data')
    ],

    [
        Input('train-btn', 'n_clicks'),
        Input('reset-btn', 'n-clicks') #added new reset button function in the callback
    ],

    [
        # follows the order of the UI
        
        #FNN architecture inputs
        State('seed-input', 'value'),
        State('hidden-layer-count', 'value'),
        State('hidden-size', 'value'), 
        State('act-dropdown', 'value'),

        #Training setup inputs
        State('optimiser-dropdown', 'value'), #gradient descent
        State('learning-rate', 'value'), 
        State('epochs', 'value'),
        State('batch-size', 'value'),
        State('reg-strength', 'value'),
        State('es-dropdown', 'value') #early stopping strategy

    ],
    prevent_initial_call=True
)
def train_visualise_or_reset(train_clicks, 
                             reset_clicks, 
                             seed, 
                             hidden_layer_count, 
                             hidden_size, 
                             activation, 
                             optimiser, 
                             learning_rate_log, 
                             epochs, 
                             batch_size, 
                             reg_strength, 
                             early_stopping):
    #train model or reset trained model
    
    ctx = callback_context
    if not ctx.triggered:
        #nothing triggered so nothing happens
        raise dash.exceptions.PreventUpdate
    
    #this added section is intended to provide the reset function
    button_id = ctx.triggered[0]['prod_id'].split('.')[0]

    #RESET method branch ==========================================
    if button_id == 'reset-btn':
        empty_fig = go.Figure()
        return (
            empty_fig, #training curves
            empty_fig, #FNN architecture 
            empty_fig, #confusion matrix
            "",        #accuracy metrics
            "",        #per class metrics
            "",        #status metrics
            None,      #model history
        )
    
    #TRAIN method branch ==========================================

    #use default values for inputs as a safeguard in the callback


    #=== FNN ARCHITECTURE CONFIGS ===

    #weight inialisation
    if seed is None:
        seed = 42
    seed = int(seed)

    #hidden layers count
    if hidden_layer_count is None:
        hidden_layer_count = 1
    hidden_layer_count = int(hidden_layer_count)

    #input layer size - match the number of features in the dataset
    if input_size is None:
        input_size = X_train.shape[1] #the iris dataset would make this 4 

    #hidden layer size - this is the value the user would change the most
    if hidden_size is None:
        hidden_size = 5
    hidden_size = int(hidden_size)

    #output layer size - match the number of classes in the dataset
    if output_size is None:
        output_size = 5
    output_size = int(output_size)

    #activation function
    if activation is None:
        activation = 'Sigmoid'

    #=== TRAINING SETUP CONFIGS ===

    #optimiser algorithm (gradient descent types)
    if optimiser is None:
        optimiser = 'Batch'

    #learning rate
    learning_rate = 10 ** learning_rate_log

    if learning_rate_log is None:
        learning_rate_log = -2

    #epochs
    if epochs is None:
        epochs = 50
    epochs = int(epochs)

    #batch size
    if batch_size is None:
        batch_size = 32
    batch_size = int(batch_size)

    #regularisation strength
    if reg_strength is None:
        reg_strength = 0.0
    reg_strength = int(reg_strength)

    #early stopping criteria
    if early_stopping is None:
        early_stopping = 'Validation Loss Plateu (Generalisation)'
   
    # Train
    model, history = train_model(X_train, y_train,
                                epochs=int(epochs),
                                learning_rate=learning_rate,
                                hidden_size=int(hidden_size),
                                seed=seed)
    
    from nn_model import SimpleNN
    np.random.seed(seed)
    val_tracking_model = SimpleNN(input_size=X_train.shape[1],
                         hidden_size=int(hidden_size),
                         output_size=3,
                         seed=seed)
    
    val_history = {'loss': [], 'accuracy': []}
    for epoch in range(int(epochs)):

        #Train in one epoch
        val_tracking_model.train_epoch(X_train, y_train, learning_rate)

        #val_model.train_epoch(X_train, y_train, learning_rate)
        val_output = val_tracking_model.forward(X_val)
        val_loss = val_tracking_model.compute_loss(val_output, y_val)
        val_preds = np.argmax(val_output, axis = 1)
        val_acc =np.mean(val_preds == y_val)
        val_history['loss'].append(val_loss)
        val_history['accuracy'].append(val_acc)
   
    # Test accuracy
    test_output = model.forward(X_test)
    test_preds = np.argmax(test_output, axis=1)
    test_acc = np.mean(test_preds == y_test)

    #confusion matrix - config ================================
    cm = confusion_matrix(y_test, test_preds, labels=[0,1,2])

    cm_fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 14},
        colorscale='Blues',
        hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra'
    ))

    cm_fig.update_layout (
        title='Test Set Confusion Matrix (per class performance)',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        height=400,
        margin=dict(l=100, r=50, t=80, b=80)
    )

   
    # Create training curves figure============================
    train_fig= go.Figure()
   
    #Training loss
    train_fig.add_trace(go.Scatter(
        y=history['loss'],
        mode='lines',
        name='Trn Loss',
        line=dict(color='#FF6B6B', width=2)
    ))

    #Validation loss (will be with a dashed line)
    train_fig.add_trace(go.Scatter(
        y=val_history['loss'],
        mode='lines',
        name='Val Loss',
        line=dict(color='#F97316', width=2, dash='dash'),
    ))

    #Training accuracy
    train_fig.add_trace(go.Scatter(
        y=history['accuracy'],
        mode='lines',
        name='Trn Acc',
        line=dict(color='#4ECDC4', width=2),
        yaxis='y2'
    ))
   
    #Validation accuracy (will be with a dashed line)
    train_fig.add_trace(go.Scatter(
        y=val_history['accuracy'],
        mode='lines',
        name='Val Acc',
        line=dict(color='#22C55E', width=2, dash='dash'),
        yaxis='y2'
    ))

   
    train_fig.update_layout(
        title='Training Progress (Solid: Training, Dashed: Validation)',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        yaxis2=dict(title='Accuracy', overlaying='y', side='right'),
        hovermode='x unified',
        height=400
    )

    #Architecture diagram figure

    input_size = X_train.shape[1]
    hidden_size_int = int(hidden_size)
    output_size = len(np.unique(y_train))

    x_input, x_hidden, x_output = 0, 1, 2

    node_x = []
    node_y = []
    node_text = []
    node_layer = []

    #Input layer nodes
    for i in range(input_size):
        node_x.append(x_input)
        node_y.append(i)
        node_text.append(f"I{i+1}")
        node_layer.append("Input")

    #Hidden layer nodes
    for i in range(hidden_size_int):
        node_x.append(x_hidden)
        node_y.append(i)
        node_text.append(f"H{i+1}")
        node_layer.append("Hidden")

    # Output layer nodes
    for i in range(output_size):
        node_x.append(x_output)
        node_y.append(i)
        node_text.append(f"O{i+1}")
        node_layer.append("Output")

    edge_x = []
    edge_y = []

    # Input → Hidden connections
    for i in range(input_size):
        for j in range(hidden_size_int):
            edge_x += [x_input, x_hidden, None]
            edge_y += [i, j, None]

    # Hidden → Output connections
    for j in range(hidden_size_int):
        for k in range(output_size):
            edge_x += [x_hidden, x_output, None]
            edge_y += [j, k, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=1, color='#BBBBBB'),
        hoverinfo='none',
        showlegend=False
    )

    node_colors = [
        '#60A5FA' if layer == 'Input'
        else '#A855F7' if layer == 'Hidden'
        else '#F97316'
        for layer in node_layer
    ]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='middle right',
        marker=dict(
            size=18,
            color=node_colors,
            line=dict(width=1, color='#333333')
        ),
        hoverinfo='text',
        showlegend=False
    )

    arch_fig= go.Figure(data=[edge_trace, node_trace])
    arch_fig.update_layout(
        title=f"Network Architecture: {input_size}–{hidden_size_int}–{output_size}",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        height=350
    )

    #Per-class metrics such as (precision, recall, f1) ================================
    precision, recall, f1, support = precision_recall_fscore_support(y_test, test_preds, labels=[0, 1, 2])

    per_class_metrics = html.Div([
        html.H4("Per-Class Performance", style={'marginBottom': 10}),
        html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Class", style={'padding': '8px', 'textAlign': 'left'}),
                    html.Th("Precision", style={'padding': '8px', 'textAlign': 'center'}),
                    html.Th("Recall", style={'padding': '8px', 'textAlign': 'center'}),
                    html.Th("F1-Score", style={'padding': '8px', 'textAlign': 'center'}),
                    html.Th("Support", style={'padding': '8px', 'textAlign': 'center'}),
                ], style={'borderBottom': '2px solid #333'})
            ),
            html.Tbody([
                html.Tr([
                    html.Td(class_names[i], style={'padding': '6px', 'fontWeight': 'bold'}),
                    html.Td(f"{precision[i]:.3f}", style={'padding': '6px', 'textAlign': 'center', 'color': '#4ECDC4'}),
                    html.Td(f"{recall[i]:.3f}", style={'padding': '6px', 'textAlign': 'center', 'color': '#22C55E'}),
                    html.Td(f"{f1[i]:.3f}", style={'padding': '6px', 'textAlign': 'center', 'color': '#F97316'}),
                    html.Td(f"{support[i]}", style={'padding': '6px', 'textAlign': 'center', 'color': '#666'}),
                ]) for i in range(3)
            ])
        ], style={
            'borderCollapse': 'collapse',
            'width': '100%',
            'fontSize': '13px',
            'border': '1px solid #ddd'
        })
    ])




    #Accuracy metrics display
    train_acc_final = history['accuracy'][-1]
    val_acc_final = val_history['accuracy'][-1]
    overfitting_gap = train_acc_final - val_acc_final

    accuracy_metrics = html.Div([
        html.P(f"Train Accuracy: {train_acc_final:.2%}", style={'color': '#4ECDC4', 'marginBottom': '5px'}),
        html.P(f"Val Accuracy: {val_acc_final:.2%}", style={'color': '#22C55E', 'marginBottom': '5px'}),
        html.P(f"Test Accuracy: {test_acc:.2%}", style={'color': '#FF6B6B', 'marginBottom': '5px', 'fontWeight': 'bold'}),
        html.P(f"Overfitting Gap: {overfitting_gap:.2%}", style={'color': '#FF6B6B' if overfitting_gap > 0.05 else 'green', 'fontSize': '12px'})
    ])
   
    status_msg = f"Training complete! Observe Outcomes (Seed={seed}, {int(hidden_size)}-neuron, LR={learning_rate:.4f})"
    #accuracy_msg = f"Test Accuracy: {test_acc:.2%}"
    return train_fig, arch_fig, cm_fig, per_class_metrics, accuracy_metrics, status_msg, {
        'train_loss': history['loss'],
        'train_acc': history['accuracy'],
        'val_loss': val_history['loss'],
        'val_acc': val_history['accuracy']
    }
   


 
 #start app
if __name__ == '__main__':
    app.run(debug=False)   #changed debug to false because otherwise it resets the page every minute


