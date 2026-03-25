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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')  # Dash-safe
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import plotly.tools as tls
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataload import load_dataset, get_dataset_stats
from trainer import train_model
from trainer import build_model

from models import SimpleNN

import io
import base64
import matplotlib
matplotlib.use('Agg') #for the png whic dash plotly can display

#models for code sandbox
#from models import SimpleNN
#from models import LogisticRegression
#from models import ComplexNN

#page layout imports
from pagelayout import home_layout
from pagelayout import skilltree_layout
from pagelayout import sandbox_layout
from pagelayout import level1_layout
from pagelayout import level2_layout
from pagelayout import level3_layout

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification #for level 2
import plotly.graph_objects as go
#endregion

#====DELCARATION: CODE HERE IS ASSISTED BY https://www.perplexity.ai/ 04/12/25 - 03/03/25====
#region datahandling
# Load data once at startup
X_train_full, X_test, y_train_full, y_test, meta = load_dataset("iris") #feature/class names removed because meta fulfills those variables

class_names = meta['class_names']
dataset_stats = get_dataset_stats(X_train_full, y_train_full) #can now make use of this with metadata

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

#endregion

#====DELCARATION: CODE HERE IS ASSISTED BY https://www.perplexity.ai/ 04/12/25 - 03/03/25====
#region SECURE PROGRAMMING (syntax map for python execution) - WORK IN PROGRESS
#safe python environment for user input
SAFE_PYTHON_ENV = {
    '__builtins__': {
    'print': print, 
    'int': int,
    'float': float, 
    'str': str,
    'list': list,
    'dict': dict, 
    'set': set,
    'range': range,
    'len': len,
    'sum': sum,
    'max': max,
    'min': min,
    '__import__': __import__ 
    },
    'pd': pd, #panda
    'np': np, #numpy library
    'plt': plt if 'plt' in globals() else None, #Matplotlib
    'from sklearn.model_selection import train_test_split': None,  # Already global
    'from sklearn.linear_model import LogisticRegression': None,
    'from sklearn.metrics import accuracy_score': None,
    'from sklearn.metrics import confusion_matrix': None,

    #x train
    'X_train': X_train, 
    'X_test': X_test, 
    'y_train': y_train, 
    'y_test': y_test,

    #full train
    'X_train_full': X_train_full, 
    'X_val': X_val, 
    'y_train_full': y_train_full, 
    'y_val': y_val,

    'load_dataset': load_dataset,
    'class_names': class_names,
    'train_model': train_model,
    #'SimpleNN': SimpleNN,
    'LogisticRegression': LogisticRegression,
    #'ComplexNN': ComplexNN,
    'accuracy_score': accuracy_score,

    'train_test_split': train_test_split,
    'confusion_matrix': confusion_matrix
}

#endregion

#====DELCARATION: CODE HERE IS ASSISTED BY https://www.perplexity.ai/ 04/12/25 - 03/03/25====
#region app setup and information layout
app = dash.Dash(__name__)

app.config.suppress_callback_exceptions = True #to prevent callback errors from the teacher page for the dataset/model selection

app.layout = html.Div([
    # Fixed top navigation bar
    html.Div([
        html.H3("Neural Network Dashboard (WORK IN PROGRESS)", 
                style={'margin': '0 20px', 'display': 'inline-block'}),
        html.Img(
        src="/assets/construction_man.gif",
        style={
            "width": "100px", 
            "height": "100px",
            "display": "block",
            "borderRadius": "50%"
        }
    ),
        html.Div([
            dcc.Link("Home", href="/home", 
                    style={'padding': '10px 15px', 'display': 'inline-block', 
                          'color': '#333', 'textDecoration': 'none'}),
            dcc.Link("Skill tree", href="/skilltree", 
                    style={'padding': '10px 15px', 'display': 'inline-block', 
                          'color': '#333', 'textDecoration': 'none'}),
            dcc.Link("Sandbox", href="/sandbox", 
                    style={'padding': '10px 15px', 'display': 'inline-block', 
                          'color': '#333', 'textDecoration': 'none'}),
            
        ], style={'float': 'right'})
    ], style={
        'backgroundColor': "#29c08e", 
        'borderBottom': '1px solid #dee2e6',
        'padding': '30px 0', 
        'position': 'sticky',
        'top': '0',
        'zIndex': '1000'
    }),


    dcc.Location(id="url", refresh=True), #url watchdog
    dcc.Store(id="lesson-config-store"),   #sharing between teacher/student of lesson config
    html.Div(id="page-content"),           #student/teacher pa
])

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
    html.H3("Predition mechanism of Feed-Forward Neural Networks (FNNs)", 
    style={"color": "#1f2937", "marginBottom": "16px"}),
    
    html.P("This dashboard provides an interactive walkthrough of a FNN solving a classification problem with the Iris dataset. \n"
    "It will provide a complementary experience of theory and tasks intended to improve conceptual understanding of Neural Network concepts."),

    html.Ul([
            html.Li("4 flower measurements × weights = hidden neuron values"),
            html.Li("Hidden neurons → activation function → class probabilities"), 
            html.Li("Live math tracing + training curves + confusion matrix")
        ], style={"fontSize": "13px", "lineHeight": "1.6", "marginBottom": "24px"}),
    
    html.Div([
        html.Span("Dataset: ", style={"fontWeight": "600"}),
        html.Span("Iris flowers (150 samples × 4 features → 3 species)"),
    ], style={"background": "#f0f9ff", "padding": "12px", "borderRadius": "6px"}),
    
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
 
#endregion

#====DELCARATION: CODE HERE IS ASSISTED BY https://www.perplexity.ai/ 04/12/25 - 03/03/25====
#region callback and method for INITAL DASHBOARD TRAINING AND VISUALISATION 
@app.callback(
    [
        Output('loss-curves', 'figure'),
        Output('accuracy-curves', 'figure'),
        Output('architecture-graph', 'figure'),
        Output('confusion-matrix-heatmap', 'figure'),
        Output('accuracy-metrics', 'children'),
        Output('per-class-metrics', 'children'),
        Output('status-text', 'children'),
        Output('model-history-store', 'data')
    ],

    [
        Input('train-btn', 'n_clicks'),
        Input('reset-btn', 'n_clicks') #added new reset button function in the callback
    ],

    [
        # follows the order of the UI

        #teacher inputs to be moved (TESTING),
        State('ds_dropdown', 'value'),
        State('model_dropdown', 'value'),
        
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
                             dataset_name,
                             model_name, 
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
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] 

    #RESET method branch ==========================================
    if button_id == 'reset-btn':
        empty_fig = go.Figure()
        return (
            empty_fig, #training curves
            empty_fig, #FNN architecture
            empty_fig, #architecture graph #new to get the reset working because it expects 8 values 
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
    
    """
    #input layer size - match the number of features in the dataset
    if input_size is None:
        input_size = X_train.shape[1] #the iris dataset would make this 4 
    """
    
    #hidden layer size - this is the value the user would change the most
    if hidden_size is None:
        hidden_size = 5
    hidden_size = int(hidden_size)

    """
    #output layer size - match the number of classes in the dataset
    if output_size is None:
        output_size = 5
    output_size = int(output_size) 
    """

    #activation function
    if activation is None:
        activation = 'Sigmoid'

    #=== TRAINING SETUP CONFIGS ===

    #optimiser algorithm (gradient descent types)
    if optimiser is None:
        optimiser = 'Batch'

    #learning rate
    if learning_rate_log is None:
        learning_rate_log = -2

    learning_rate = 10 ** float(learning_rate_log)

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
   
    #1. Build model

    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))

    model_map = { #this is to check that the dropdown labels selected go to internal keys in order to build the actual model
        "Logistic Regression": "log_reg",
        "NN-1-Layer": "simple_nn",
        "NN-2-Layer": "two_layer_nn"
    }

    model_key = model_map.get(model_name)

    if model_key is None: #will display unknown model with actual bad model name
        #to fail elegantly if there's no model
        status_msg = f"Unknown model selection: {model_name}"
        empty_fig = go.Figure()
        return(
            empty_fig, empty_fig, empty_fig, empty_fig,
            "", "", status_msg, None
        )
    
    #print("model_name from dropdown:", model_name)
    #print("model_key for build_model:", model_key)


    model = build_model(
        model_name=model_key, #will pass the interal key ("simple_nn") instead of the UI label
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        seed=seed,
    )

    #2. train model

    model, history = train_model(model, 
                                 X_train, 
                                 y_train,
                                epochs=int(epochs),
                                learning_rate=learning_rate,
                                )
    
    from models import SimpleNN
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

   
    #loss convergence curves figure=============================
    def build_loss_fig(history, val_history):
        fig = go.Figure()

        #training loss to show convergence over epochs
        fig.add_trace(go.Scatter(
            y=history['loss'],
            mode='lines',
            name='Training Loss',
            line=dict(color='#FF6B6B', width=3),
            hovertemplate='Epoch: %{x}<br>Loss: %{y: .4f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            y=val_history['loss'],
            mode='lines',
            name='Validation Loss',
            line=dict(color='#F97316', width=3, dash='dash'),
            hovertemplate='Epoch: %{x}<br>Loss: %{y: .4f}<extra></extra>'
        ))

        fig.update_layout(
            title='Loss Convergence (Training vs Validation)',
            xaxis_title='Epoch',
            yaxis_title='Cross Loss', #entropy?
            hovermode='x unified',
            height=2000,
            showlegend=True,
            legend=dict(x=0, y=1)
        )
        return fig
    #accuracy generalisation curves figure======================
    def build_accuracy_fig(history, val_history):
        fig = go.Figure()

        #training accuracy to show generalisation
        fig.add_trace(go.Scatter(
            y=history['accuracy'],
            mode='lines',
            name='Training Accuracy',
            line=dict(color='#4ECDC4', width=3),
            hovertemplate='Epoch: %{x}<br>Acc: %{y: .1%}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            y=val_history['accuracy'],
            mode='lines',
            name='Validation Accuracy',
            line=dict(color='#F97316', width=3, dash='dash'),
            hovertemplate='Epoch: %{x}<br>Acc: %{y: .1%}<extra></extra>'
        ))

        fig.update_layout(
            title='Generalisation Gap (Training vs Validation)',
            xaxis_title='Epoch',
            yaxis_title='Accuracy', 
            hovermode='x unified',
            height=2000,
            showlegend=True,
            legend=dict(x=0, y=1)
        )
        return fig

    #Architecture diagram figure =============================

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

    # Input => Hidden connections
    for i in range(input_size):
        for j in range(hidden_size_int):
            edge_x += [x_input, x_hidden, None]
            edge_y += [i, j, None]

    # Hidden => Output connections
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

    loss_fig = build_loss_fig(history, val_history)
    accuracy_fig = build_accuracy_fig(history, val_history)
   
    status_msg = f"Training complete! Observe Outcomes (Seed={seed}, {int(hidden_size)}-neuron, LR={learning_rate:.4f})"
    #accuracy_msg = f"Test Accuracy: {test_acc:.2%}"
    return loss_fig, accuracy_fig, arch_fig, cm_fig, per_class_metrics, accuracy_metrics, status_msg, {
        'train_loss': history['loss'],
        'train_acc': history['accuracy'],
        'val_loss': val_history['loss'],
        'val_acc': val_history['accuracy']
    }

#endregion   

#====DELCARATION: CODE HERE IS ASSISTED BY https://www.perplexity.ai/ 04/12/25 - 03/03/25====
#region page layout routing logic (callbacks/ methods)
#callback for display decision: student or teacher page
#now the callback to get to the homepage actually works, can navigate all pages
@app.callback(
   Output("page-content", "children"),
   Input("url", "pathname"), 
)
def display_decision(pathname): #this is a basic page selector before it gets transferred to the skill tree
    if pathname == "/home":
        return home_layout()
    elif pathname == "/skilltree":
        return skilltree_layout()
    elif pathname == "/sandbox":
        return sandbox_layout()
    elif pathname == "/level1":
        return level1_layout()
    elif pathname == "/level2":
        return level2_layout()
    elif pathname == "/level3":
        return level3_layout()
    elif pathname == "/level4":
        return level4_layout()
    elif pathname == "/level5":
        return level5_layout()
    else:
        return home_layout()

#endregion

#====DELCARATION: CODE HERE IS ASSISTED BY https://www.perplexity.ai/ 04/12/25 - 03/03/25====
#region SECURE PROGRAMMING (callback for interpreter control, python execution method and syntax highlighter) - WORK IN PROGRESS

#callback for INTERPRETER CONTROL
#this will be simpler because this app is already using a python interpreter
#just need to figure how this works for external python code?
@app.callback(
    [Output("editor-output", "children"),
     Output("editor-error", "children"),
     Output("editor-plot", "figure"),
     Output("download-editor", "data")],
    [Input("code-run", "n_clicks"),
     Input("code-export", "n_clicks")],
     State("code-input", "value"),

    prevent_initial_call=True    
)

def execute_python_code(run_clicks, export_clicks, user_code):
    # EXPORT BUTTON
    if export_clicks:
        return "", "", go.Figure(), dict(content=user_code, filename="script.py")
    
    # EXECUTE BUTTON
    import io
    output_capture = io.StringIO()
    
    try:
        # CAPTURE PRINT OUTPUT
        import sys
        old_stdout = sys.stdout
        sys.stdout = output_capture
        
        # SAFE EXECUTION (no dangerous builtins!)
        exec(user_code, SAFE_PYTHON_ENV)
        
        sys.stdout = old_stdout
        result = output_capture.getvalue()

        outputs = []

        if plt.get_fignums():
            fig = plt.gcf()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close('all')
            outputs.append(html.Img(
                src=f"data:image/png;base64,{img_str}",
                style={'max-width': '100%', 'height': 'auto', 'display': 'block', 'margin-bottom': '10px'}
            ))

        # Print output immediately follows
        if result.strip():
            outputs.append(html.Pre(
                result,
                style={'margin': '0', 'white-space': 'pre-wrap', 'font-family': 'monospace', 'font-size': '14px'}
            ))

        # Empty result → "success"
        if not outputs:
            outputs.append(html.Pre("Code executed successfully (no output)"))
        
        return html.Div(outputs), "", go.Figure(), dash.no_update

        
    except SyntaxError as e:
        return (
            "",
            html.Div([
                html.H3(f"Syntax Error (Line {e.lineno})", style={"color": "red"}),
                html.Pre(str(e))
            ]),
            go.Figure(),
            dash.no_update
        )
    except Exception as e:
        return (
            "",
            html.Div([
                html.H3("Compile Error", style={"color": "red"}),
                html.Pre(str(e))
            ]),
            go.Figure(),
            dash.no_update
        )


#callback for a custom syntax highlighter/validator to read python code from the user
#how will i get the error message if there's a compile fail?
@app.callback(
    Output("code-highlighted", "children"),
    Input("code-input", "value"),
)

def syntax_highlighter(pythonCode):
    if not pythonCode: #if there's no code in the text area it must be returned as an empty value
        return ""
    
    #the following highlighter needs to be inline and using regex principles
    #it will go character by character, no need to look at blank spaces unless necessary
    #will this method include validation??

#endregion

#====DELCARATION: CODE HERE IS ASSISTED BY Copilot (GPT-5.4) 22/03/26 - 23/04/26====
#region MODEL FACTORY: Level 1 callbacks/methods (linear classifiers and 2d decision boundary visualisation) - WORK IN PROGRESS

# callback to show 2d decision boundary which dynamically changes with hidden layer slider
@app.callback(
    [Output('live-architecture', 'figure'), Output('live-weights-heatmap', 'figure'),
     Output('code-preview', 'children')],
    [Input('live-hidden-size', 'value'), Input('live-seed', 'value')]
)
def update_live_visualisations(hidden_size, seed):
    # FIX 1: Load Iris data locally (no global dependency)
    from dataload import loaddataset
    Xtrain, _, ytrain, _, meta = loaddataset('iris')
    
    # FIX 2: Build model (no Xtrain needed for architecture viz)
    import numpy as np
    np.random.seed(seed or 42)
    model = SimpleNN(4, hidden_size, 3)  # Iris: 4 features → 3 classes
    
    # 3. ARCHITECTURE DIAGRAM (works without training)
    arch_fig = build_architecture_diagram(model, hidden_size)
    
    # 4. WEIGHT HEATMAPS 
    weight_fig = go.Figure()
    weight_fig.add_trace(go.Heatmap(z=model.W1, colorscale='RdBu', zmid=0))
    weight_fig.update_layout(title=f"W1 Weights (4×{hidden_size})")
    
    # 5. CODE PREVIEW
    code_preview = html.Textarea(
        value=f"model = SimpleNN(4, {hidden_size}, 3, seed={seed})",
        style={'width': '100%', 'height': 100}
    )
    
    return arch_fig, weight_fig, code_preview

def build_architecture_diagram(model, hidden_size):
    fig = go.Figure()
    
    # Nodes: Input(4) → Hidden(N) → Output(3)
    x_nodes = [0, 1, 2]  # Layers
    input_nodes = [i/5 for i in range(4)]
    hidden_nodes = [i/16 for i in range(hidden_size)]  
    output_nodes = [i/3 for i in range(3)]
    
    # Add nodes
    fig.add_trace(go.Scatter(x=[0]*4, y=input_nodes, mode='markers+text',
                           marker=dict(size=20, color='blue'), 
                           text=[f'I{i+1}' for i in range(4)],
                           name='Input'))
    fig.add_trace(go.Scatter(x=[1]*hidden_size, y=hidden_nodes, mode='markers+text',
                           marker=dict(size=15, color='orange'),
                           text=[f'H{i+1}' for i in range(hidden_size)],
                           name='Hidden'))
    fig.add_trace(go.Scatter(x=[2]*3, y=output_nodes, mode='markers+text',
                           marker=dict(size=20, color='green'),
                           text=['O1','O2','O3'], name='Output'))
    
    # Edges (all connections)
    for i in range(4):
        for j in range(hidden_size):
            fig.add_trace(go.Scatter(x=[0,1], y=[input_nodes[i], hidden_nodes[j]], 
                                   mode='lines', line=dict(width=1, color='gray'),
                                   showlegend=False, hoverinfo='skip'))
    
    fig.update_layout(title=f"Live Architecture: 4 → {hidden_size} → 3",
                     xaxis=dict(showgrid=False, range=[-0.2, 2.2]),
                     yaxis=dict(showgrid=False, range=[-0.2, 1.2]),
                     height=400, showlegend=False)
    return fig

# a self contained boundary decision for level 1
from sklearn.decomposition import PCA

def plot_decision_boundary(model, Xtrain_sample):
    """Req 3.1.5: Live decision boundary from current model params"""
    # PCA to 2D for visualization (Iris: 4→2 dims)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(Xtrain_sample[:100])  # Sample for speed
    
    # Predict on 2D grid
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on grid (model expects 4D input → pad with zeros)
    grid_input = np.c_[xx.ravel(), yy.ravel(), np.zeros((xx.ravel().shape[0], 2))]
    Z = model.forward(grid_input)  # Your SimpleNN.forward()
    Z = np.argmax(Z, axis=1).reshape(xx.shape)
    
    # Plot
    fig = go.Figure(data=go.Heatmap(
        z=Z, x=xx[0], y=yy[:,0], colorscale='RdYlBu', hoverongaps=False
    ))
    fig.update_layout(title="Live Decision Boundary (PCA Projection)",
                      xaxis_title="PC1", yaxis_title="PC2", height=350)
    return fig

#callback for 2d decision boundary of level 1
def generate_data(dataset: str):
    """
    Return (X, y) for a 2D toy dataset.

    X: (n_samples, 2), y: (n_samples,)
    """
    if dataset == 'linear':
        # Two linearly separable blobs
        X, y = make_blobs(
            n_samples=300,
            centers=[(-2, -2), (2, 2)],
            cluster_std=0.8,
            random_state=42
        )
    elif dataset == 'moons':
        X, y = make_moons(
            n_samples=300,
            noise=0.2,
            random_state=42
        )  # two interleaving half-circles[web:55][web:59]
    elif dataset == 'circles':
        X, y = make_circles(
            n_samples=300,
            noise=0.1,
            factor=0.4,
            random_state=42
        )  # concentric circles[web:53][web:59]
    else:
        # Fallback: simple blobs
        X, y = make_blobs(
            n_samples=300,
            centers=[(-2, -2), (2, 2)],
            cluster_std=0.8,
            random_state=42
        )

    return X, y

@app.callback(
    Output('l1-decision-boundary', 'figure'),
    [
        Input('l1-dataset', 'value'),
        Input('l1-w1', 'value'),
        Input('l1-w2', 'value'),
        Input('l1-bias', 'value'),
    ]
)
def update_decision_boundary(dataset, w1, w2, b):
    # 1. Get data
    X, y = generate_data(dataset)  # X: (n, 2), y: (n,)
    x1 = X[:, 0]
    x2 = X[:, 1]

    # 2. Base scatter plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x1[y == 0],
            y=x2[y == 0],
            mode='markers',
            name='Class 0',
            marker=dict(color='blue', size=8, opacity=0.7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x1[y == 1],
            y=x2[y == 1],
            mode='markers',
            name='Class 1',
            marker=dict(color='red', size=8, opacity=0.7),
        )
    )

    # 3. Decision boundary line (if w2 != 0)
    x_min, x_max = x1.min() - 0.5, x1.max() + 0.5

    if abs(w2) > 1e-6:
        xs = np.linspace(x_min, x_max, 100)
        ys = -(w1 * xs + b) / w2

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode='lines',
                name='Decision boundary',
                line=dict(color='black', width=2),
            )
        )

    fig.update_layout(
        xaxis_title='x₁',
        yaxis_title='x₂',
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(scaleanchor='x', scaleratio=1),  # keep aspect ratio square
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=10, t=40, b=40),
    )

    return fig

#endregion

#====DECLARATION: CODE HERE IS ASSISTED BY Copilot (GPT-5.4) 22/03/26 - 23/04/26====
#region MODEL FACTORY: Level 2 callbacks/methods to visual the structure of a single hidde layer perception with decision boundary - WORK IN PROGRESS
#toy datasets 
def load_toy_dataset(name, n_samples=300, noise=0.2, random_state=0):
    if name == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif name == 'circles':
        X, y = make_circles(n_samples=n_samples, factor=0.5,
                            noise=noise, random_state=random_state)
    elif name == 'linear':
        # simple linearly separable classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=1.5,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Ensure shapes (N, 2) and (N,)
    return X.astype(np.float32), y.astype(np.int32)

def init_level2_mlp(input_dim=2, hidden_layers=None, output_dim=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if hidden_layers is None:
        hidden_layers = [6, 6]

    layer_sizes = [input_dim] + list(hidden_layers) + [output_dim]
    weights = []
    biases = []

    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        weights.append(
            rng.normal(0.0, 1.0 / np.sqrt(fan_in), size=(fan_out, fan_in))
        )
        biases.append(np.zeros((fan_out, 1)))

    return {
        'weights': [weight.tolist() for weight in weights],
        'biases': [bias.tolist() for bias in biases],
        'epoch': 0,
        'history': {
            'epochs': [0],
            'loss': [],
            'accuracy': [],
        },
    }


def level2_deserialize_params(params):
    weights = [np.array(weight, dtype=np.float64) for weight in params['weights']]
    biases = [np.array(bias, dtype=np.float64) for bias in params['biases']]
    return weights, biases


def level2_serialize_params(weights, biases, params):
    params['weights'] = [weight.tolist() for weight in weights]
    params['biases'] = [bias.tolist() for bias in biases]
    return params


def level2_parameter_count(params):
    weights, biases = level2_deserialize_params(params)
    return int(sum(weight.size + bias.size for weight, bias in zip(weights, biases)))

def activation_forward(Z, activation):
    if activation == 'relu':
        A = np.maximum(0, Z)
    elif activation == 'tanh':
        A = np.tanh(Z)
    elif activation == 'sigmoid':
        A = 1.0 / (1.0 + np.exp(-Z))
    else:
        raise ValueError(f"Unknown activation: {activation}")
    return A

def activation_backward(dA, Z, activation):
    if activation == 'relu':
        dZ = dA * (Z > 0)
    elif activation == 'tanh':
        A = np.tanh(Z)
        dZ = dA * (1 - A**2)
    elif activation == 'sigmoid':
        A = 1.0 / (1.0 + np.exp(-Z))
        dZ = dA * A * (1 - A)
    else:
        raise ValueError(f"Unknown activation: {activation}")
    return dZ

def level2_forward_pass(X, params, activation):
    """Run a configurable MLP forward pass for the Level 2 builder."""
    weights, biases = level2_deserialize_params(params)
    activations = [X.T]
    pre_activations = []
    current_activation = X.T

    for weight, bias in zip(weights[:-1], biases[:-1]):
        Z = weight @ current_activation + bias
        pre_activations.append(Z)
        current_activation = activation_forward(Z, activation)
        activations.append(current_activation)

    Z_out = weights[-1] @ current_activation + biases[-1]
    A_out = 1.0 / (1.0 + np.exp(-Z_out))
    pre_activations.append(Z_out)
    activations.append(A_out)

    return A_out, {
        'activations': activations,
        'pre_activations': pre_activations,
    }


def level2_evaluate_metrics(X, y, params, activation, l2=0.0):
    predictions, _ = level2_forward_pass(X, params, activation)
    y_row = y.reshape(1, -1)
    eps = 1e-8
    loss = -np.mean(y_row * np.log(predictions + eps) + (1 - y_row) * np.log(1 - predictions + eps))

    if l2 > 0:
        weights, _ = level2_deserialize_params(params)
        loss += 0.5 * l2 * sum(np.sum(weight * weight) for weight in weights)

    accuracy = np.mean((predictions >= 0.5).astype(np.int32) == y_row)
    return {
        'loss': float(loss),
        'accuracy': float(accuracy),
        'epoch': int(params.get('epoch', 0)),
        'parameter_count': level2_parameter_count(params),
    }


def level2_set_baseline_history(X, y, params, activation, l2=0.0):
    metrics = level2_evaluate_metrics(X, y, params, activation, l2=l2)
    params['epoch'] = 0
    params['history'] = {
        'epochs': [0],
        'loss': [metrics['loss']],
        'accuracy': [metrics['accuracy']],
    }
    return params


def train_level2_model(X, y, params, activation='tanh', epochs=80, lr=0.08, l2=1e-4):
    """Train the Level 2 configurable MLP with full-batch gradient descent."""
    weights, biases = level2_deserialize_params(params)
    y_row = y.reshape(1, -1)
    sample_count = X.shape[0]

    history = params.get('history', {'epochs': [0], 'loss': [], 'accuracy': []})
    epochs_history = list(history.get('epochs', [0]))
    loss_history = list(history.get('loss', []))
    accuracy_history = list(history.get('accuracy', []))

    for _ in range(epochs):
        predictions, cache = level2_forward_pass(
            X,
            {'weights': [weight.tolist() for weight in weights], 'biases': [bias.tolist() for bias in biases]},
            activation
        )
        dZ = predictions - y_row
        gradients_w = [None] * len(weights)
        gradients_b = [None] * len(biases)

        for layer_index in reversed(range(len(weights))):
            prev_activation = cache['activations'][layer_index]
            gradients_w[layer_index] = (dZ @ prev_activation.T) / sample_count
            gradients_b[layer_index] = np.mean(dZ, axis=1, keepdims=True)

            if l2 > 0:
                gradients_w[layer_index] += l2 * weights[layer_index]

            if layer_index > 0:
                dA_prev = weights[layer_index].T @ dZ
                dZ = activation_backward(
                    dA_prev,
                    cache['pre_activations'][layer_index - 1],
                    activation
                )

        for layer_index in range(len(weights)):
            weights[layer_index] -= lr * gradients_w[layer_index]
            biases[layer_index] -= lr * gradients_b[layer_index]

        params['epoch'] = int(params.get('epoch', 0)) + 1
        serialized_params = {
            'weights': [weight.tolist() for weight in weights],
            'biases': [bias.tolist() for bias in biases],
            'epoch': params['epoch'],
        }
        metrics = level2_evaluate_metrics(X, y, serialized_params, activation, l2=l2)
        epochs_history.append(params['epoch'])
        loss_history.append(metrics['loss'])
        accuracy_history.append(metrics['accuracy'])

    params = level2_serialize_params(weights, biases, params)
    params['history'] = {
        'epochs': epochs_history,
        'loss': loss_history,
        'accuracy': accuracy_history,
    }
    return params

def make_decision_boundary_figure(X, y, params, activation, grid_step=0.03):
    # Bounds
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step),
        np.arange(y_min, y_max, grid_step)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Forward pass on grid
    A2_grid, _ = level2_forward_pass(grid_points, params, activation)
    Z = A2_grid.reshape(xx.shape)  # predicted probability

    contour = go.Contour(
        x=xx[0, :],
        y=yy[:, 0],
        z=Z,
        showscale=False,
        contours=dict(showlines=False),
        colorscale='RdBu',
        opacity=0.6
    )

    scatter = go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(
            color=y,
            colorscale='Viridis',
            line=dict(width=1, color='black'),
            size=7
        ),
        name='Data'
    )

    fig = go.Figure(data=[contour, scatter])
    fig.update_layout(
        title="Decision boundary in input space",
        xaxis_title="x₁",
        yaxis_title="x₂",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def make_activation_figure(activation):
    z = np.linspace(-5, 5, 400)
    if activation == 'relu':
        a = np.maximum(0, z)
        title = "ReLU activation ρ(z) = max(0, z)"
    elif activation == 'tanh':
        a = np.tanh(z)
        title = "Tanh activation ρ(z) = tanh(z)"
    elif activation == 'sigmoid':
        a = 1.0 / (1.0 + np.exp(-z))
        title = "Sigmoid activation ρ(z) = 1 / (1 + e^{-z})"
    else:
        a = z
        title = f"Unknown activation: {activation}"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z, y=a, mode='lines'))
    fig.update_layout(
        title=title,
        xaxis_title="z",
        yaxis_title="ρ(z)",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def make_network_diagram_figure(input_dim=2, hidden_layers=None, output_dim=1,
                                params=None, activation='tanh'):
    if hidden_layers is None:
        hidden_layers = [6, 6]

    layer_sizes = [input_dim] + list(hidden_layers) + [output_dim]
    layer_x_positions = np.linspace(0, 1, len(layer_sizes))
    weights, biases = level2_deserialize_params(params) if params else (None, None)

    nodes_x = []
    nodes_y = []
    labels = []
    hover_text = []
    colors = []
    edge_traces = []
    layer_coordinates = []
    annotations = []

    for layer_index, layer_size in enumerate(layer_sizes):
        y_positions = np.linspace(0, 1, layer_size)
        layer_coordinates.append(y_positions)
        if layer_index == 0:
            layer_name = 'Input'
            color = '#93c5fd'
        elif layer_index == len(layer_sizes) - 1:
            layer_name = 'Output'
            color = '#fca5a5'
        else:
            layer_name = f'Hidden {layer_index}'
            color = '#86efac'

        annotations.append(dict(
            x=layer_x_positions[layer_index],
            y=1.1,
            text=f"{layer_name}<br>{layer_size} neuron(s)",
            showarrow=False,
            font=dict(size=11)
        ))

        for node_index, y_position in enumerate(y_positions):
            nodes_x.append(layer_x_positions[layer_index])
            nodes_y.append(y_position)
            colors.append(color)

            if layer_index == 0:
                labels.append(f"x{node_index + 1}")
                hover_text.append(f"Input feature x{node_index + 1}")
            elif layer_index == len(layer_sizes) - 1:
                labels.append('ŷ')
                bias_value = biases[layer_index - 1][node_index, 0] if biases is not None else 0.0
                hover_text.append(
                    f"Output node<br>bias={bias_value:+.3f}<br>ŷ = σ(z)"
                )
            else:
                labels.append(f"h{layer_index}.{node_index + 1}")
                bias_value = biases[layer_index - 1][node_index, 0] if biases is not None else 0.0
                hover_text.append(
                    f"Layer {layer_index} neuron {node_index + 1}<br>bias={bias_value:+.3f}<br>a = {activation}(z)"
                )

    for layer_index in range(len(layer_sizes) - 1):
        source_x = layer_x_positions[layer_index]
        target_x = layer_x_positions[layer_index + 1]
        source_y_positions = layer_coordinates[layer_index]
        target_y_positions = layer_coordinates[layer_index + 1]
        current_weights = weights[layer_index] if weights is not None else None

        for source_index, source_y in enumerate(source_y_positions):
            for target_index, target_y in enumerate(target_y_positions):
                if current_weights is not None:
                    weight_value = current_weights[target_index, source_index]
                    edge_color = 'steelblue' if weight_value >= 0 else 'crimson'
                    edge_width = max(0.5, min(4.5, abs(weight_value) * 2.5))
                    edge_hover = (
                        f"Layer {layer_index + 1} weight[{target_index + 1},{source_index + 1}] = "
                        f"{weight_value:+.3f}"
                    )
                else:
                    edge_color = 'grey'
                    edge_width = 1
                    edge_hover = 'Weight'

                edge_traces.append(go.Scatter(
                    x=[source_x, (source_x + target_x) / 2, target_x],
                    y=[source_y, (source_y + target_y) / 2, target_y],
                    mode='lines',
                    line=dict(color=edge_color, width=edge_width),
                    hovertext=[None, edge_hover, None],
                    hoverinfo='text',
                    showlegend=False
                ))

    node_trace = go.Scatter(
        x=nodes_x,
        y=nodes_y,
        mode='markers+text',
        text=labels,
        textposition='middle right',
        hovertext=hover_text,
        hoverinfo='text',
        marker=dict(size=16, color=colors, line=dict(width=1, color='black'))
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title='User-built network architecture',
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=0, r=0, t=70, b=0),
        showlegend=False,
        annotations=annotations,
    )
    return fig


def make_level2_training_curves_figure(history):
    epochs = history.get('epochs', [])
    losses = history.get('loss', [])
    accuracies = history.get('accuracy', [])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=losses,
        mode='lines+markers',
        name='Loss',
        line=dict(color='#dc2626', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=epochs,
        y=accuracies,
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='#0f766e', width=3),
        yaxis='y2'
    ))
    fig.update_layout(
        title='Optimisation behaviour across training runs',
        xaxis_title='Epoch',
        yaxis=dict(title='Loss'),
        yaxis2=dict(title='Accuracy', overlaying='y', side='right', range=[0, 1]),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def make_level2_metrics_cards(metrics):
    card_style = {
        'backgroundColor': 'white',
        'borderRadius': '14px',
        'padding': '14px 16px',
        'boxShadow': '0 2px 8px rgba(15, 23, 42, 0.08)',
        'border': '1px solid #e5e7eb'
    }
    label_style = {'fontSize': '12px', 'textTransform': 'uppercase', 'color': '#64748b', 'letterSpacing': '0.08em'}
    value_style = {'fontSize': '24px', 'fontWeight': '700', 'marginTop': '6px'}
    subtitle_style = {'fontSize': '12px', 'color': '#64748b', 'marginTop': '4px'}

    return [
        html.Div([
            html.Div('Accuracy', style=label_style),
            html.Div(f"{metrics['accuracy'] * 100:.1f}%", style=value_style),
            html.Div('Classification performance on the selected toy dataset.', style=subtitle_style),
        ], style=card_style),
        html.Div([
            html.Div('Loss', style=label_style),
            html.Div(f"{metrics['loss']:.4f}", style=value_style),
            html.Div('Binary cross-entropy after the current training run.', style=subtitle_style),
        ], style=card_style),
        html.Div([
            html.Div('Parameters', style=label_style),
            html.Div(str(metrics['parameter_count']), style=value_style),
            html.Div('Trainable weights and biases in the current builder architecture.', style=subtitle_style),
        ], style=card_style),
        html.Div([
            html.Div('Epoch', style=label_style),
            html.Div(str(metrics['epoch']), style=value_style),
            html.Div('Accumulated full-batch training epochs for this run.', style=subtitle_style),
        ], style=card_style),
    ]


def make_level2_summary_panel(params, activation):
    meta = params.get('meta', {})
    hidden_layers = meta.get('hidden_layer_sizes', [6, 6])
    layer_sizes = [2] + list(hidden_layers) + [1]
    weights, biases = level2_deserialize_params(params)
    layer_items = []

    for layer_index, (weight, bias) in enumerate(zip(weights, biases), start=1):
        if layer_index < len(weights):
            layer_label = f"Hidden {layer_index}"
            transform = activation
        else:
            layer_label = 'Output'
            transform = 'sigmoid'

        layer_items.append(
            html.Li(
                f"{layer_label}: W{layer_index} shape {weight.shape}, b{layer_index} shape {bias.shape}, activation={transform}",
                style={'fontFamily': 'monospace', 'fontSize': '11px'}
            )
        )

    return html.Div([
        html.Div(
            f"Architecture: {' → '.join(str(size) for size in layer_sizes)}",
            style={'fontWeight': '700', 'marginBottom': '10px'}
        ),
        html.P(
            "Forward map: h⁽ˡ⁾ = ρ(W⁽ˡ⁾h⁽ˡ⁻¹⁾ + b⁽ˡ⁾), with a sigmoid output layer for binary classification.",
            style={'fontSize': '13px', 'marginBottom': '10px'}
        ),
        html.P(
            f"Hidden activation: {activation} | Total parameters: {level2_parameter_count(params)}",
            style={'fontSize': '13px', 'color': '#475569'}
        ),
        html.Ul(layer_items, style={'paddingLeft': '18px', 'marginBottom': '10px'}),
        html.Div(
            "Tip: add layers or neurons to increase expressivity, then compare whether the extra capacity actually improves the learned boundary and training curves.",
            style={'fontSize': '12px', 'color': '#64748b'}
        )
    ])


def make_level2_comparison_panel(compare_store, current_metrics):
    if not compare_store:
        return html.Div([
            html.H4('Comparison Run', style={'marginTop': '0'}),
            html.P(
                'Save a baseline with Compare Run, then change the architecture or train again to inspect the difference in accuracy, loss, and model size.',
                style={'fontSize': '13px', 'color': '#64748b', 'marginBottom': '0'}
            ),
        ])

    saved_metrics = compare_store['metrics']
    saved_meta = compare_store['meta']
    accuracy_delta = current_metrics['accuracy'] - saved_metrics['accuracy']
    loss_delta = current_metrics['loss'] - saved_metrics['loss']
    parameter_delta = current_metrics['parameter_count'] - saved_metrics['parameter_count']

    return html.Div([
        html.H4('Comparison Run', style={'marginTop': '0'}),
        html.P(
            f"Saved baseline: {saved_meta['dataset']} | {' → '.join(str(size) for size in saved_meta['layer_sizes'])} | {saved_meta['activation']}",
            style={'fontSize': '13px', 'marginBottom': '10px'}
        ),
        html.Div(f"Accuracy delta: {accuracy_delta * 100:+.1f}%", style={'fontWeight': '600', 'marginBottom': '6px'}),
        html.Div(f"Loss delta: {loss_delta:+.4f}", style={'fontWeight': '600', 'marginBottom': '6px'}),
        html.Div(f"Parameter delta: {parameter_delta:+d}", style={'fontWeight': '600', 'marginBottom': '10px'}),
        html.P(
            'Use this panel to decide whether added capacity improved general behaviour or only made optimisation heavier.',
            style={'fontSize': '12px', 'color': '#64748b', 'marginBottom': '0'}
        ),
    ])

@app.callback(
    Output('level2-params-store', 'data'),
    Output('level2-compare-store', 'data'),
    Input('level2-train-btn', 'n_clicks'),
    Input('level2-reset-btn', 'n_clicks'),
    Input('level2-compare-btn', 'n_clicks'),
    Input('level2-hidden-layers-slider', 'value'),
    Input('level2-neurons-slider', 'value'),
    Input('level2-activation-dropdown', 'value'),
    Input('level2-dataset-dropdown', 'value'),
    State('level2-params-store', 'data'),
    State('level2-compare-store', 'data'),
)
def update_level2_params(n_train, n_reset, n_compare, hidden_layers, neurons_per_layer,
                         activation, dataset, params, compare_store):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    hidden_layer_sizes = [neurons_per_layer] * hidden_layers
    meta = {
        'hidden_layers': hidden_layers,
        'neurons_per_layer': neurons_per_layer,
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'dataset': dataset,
        'layer_sizes': [2] + hidden_layer_sizes + [1],
    }
    rebuild_triggers = {
        None,
        'level2-reset-btn',
        'level2-hidden-layers-slider',
        'level2-neurons-slider',
        'level2-activation-dropdown',
        'level2-dataset-dropdown',
    }

    if params is None or trigger in rebuild_triggers:
        params = init_level2_mlp(input_dim=2, hidden_layers=hidden_layer_sizes, output_dim=1)
        params['meta'] = meta
        X, y = load_toy_dataset(dataset)
        params = level2_set_baseline_history(X, y, params, activation, l2=1e-4)

    if trigger == 'level2-train-btn':
        params['meta'] = meta
        X, y = load_toy_dataset(dataset)
        params = train_level2_model(X, y, params, activation=activation, epochs=80, lr=0.08, l2=1e-4)

    if trigger == 'level2-compare-btn':
        params['meta'] = meta
        X, y = load_toy_dataset(dataset)
        metrics = level2_evaluate_metrics(X, y, params, activation, l2=1e-4)
        compare_store = {
            'meta': meta,
            'metrics': metrics,
        }

    params['meta'] = meta
    return params, compare_store

@app.callback(
    Output('level2-decision-boundary-graph', 'figure'),
    Output('level2-activation-graph', 'figure'),
    Output('level2-network-diagram-graph', 'figure'),
    Output('level2-math-explanation', 'children'),
    Output('level2-metrics-row', 'children'),
    Output('level2-comparison-panel', 'children'),
    Output('level2-training-curves-graph', 'figure'),
    Input('level2-params-store', 'data'),
    Input('level2-compare-store', 'data')
)
def update_level2_views(params, compare_store):
    if params is None:
        raise dash.exceptions.PreventUpdate

    meta = params.get('meta', {})
    hidden_layer_sizes = meta.get('hidden_layer_sizes', [6, 6])
    activation = meta.get('activation', 'tanh')
    dataset = meta.get('dataset', 'moons')

    X, y = load_toy_dataset(dataset)
    metrics = level2_evaluate_metrics(X, y, params, activation, l2=1e-4)

    fig_boundary = make_decision_boundary_figure(X, y, params, activation)
    fig_boundary.update_layout(
        title=(
            f"{dataset.title()} dataset | Architecture {' → '.join(str(size) for size in meta.get('layer_sizes', [2] + hidden_layer_sizes + [1]))}"
        )
    )

    fig_activation = make_activation_figure(activation)
    fig_network = make_network_diagram_figure(
        input_dim=2,
        hidden_layers=hidden_layer_sizes,
        output_dim=1,
        params=params, activation=activation
    )
    explanation = make_level2_summary_panel(params, activation)
    metric_cards = make_level2_metrics_cards(metrics)
    comparison_panel = make_level2_comparison_panel(compare_store, metrics)
    curves_figure = make_level2_training_curves_figure(params.get('history', {}))

    return (
        fig_boundary,
        fig_activation,
        fig_network,
        explanation,
        metric_cards,
        comparison_panel,
        curves_figure,
    )

#endregion

#====DECLARATION: CODE HERE IS ASSISTED BY Copilot (GPT-5.4) 22/03/26 - 23/04/26====
#region MODEL FACTORY: Level 3 callbacks/methods to illustrate a deeper network structure (why is depth helpful) - WORK IN PROGRESS

def make_level3_placeholder_figure(title, message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref='paper',
        yref='paper',
        showarrow=False,
        font=dict(size=14, color='#475569')
    )
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='#f8fafc',
        paper_bgcolor='white'
    )
    return fig


def level3_build_meta(dataset, depth, width, activation, epochs):
    hidden_layer_sizes = [width] * depth
    return {
        'dataset': dataset,
        'depth': depth,
        'width': width,
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': activation,
        'epochs': epochs,
        'layer_sizes': [2] + hidden_layer_sizes + [1],
    }


def level3_serialize_split(X_train, X_test, y_train, y_test, X_full, y_full):
    return {
        'X_train': X_train.tolist(),
        'X_test': X_test.tolist(),
        'y_train': y_train.tolist(),
        'y_test': y_test.tolist(),
        'X_full': X_full.tolist(),
        'y_full': y_full.tolist(),
    }


def level3_deserialize_split(data):
    return (
        np.array(data['X_train'], dtype=np.float64),
        np.array(data['X_test'], dtype=np.float64),
        np.array(data['y_train'], dtype=np.int32),
        np.array(data['y_test'], dtype=np.int32),
        np.array(data['X_full'], dtype=np.float64),
        np.array(data['y_full'], dtype=np.int32),
    )


def level3_initialize_store(meta):
    X_full, y_full = load_toy_dataset(meta['dataset'])
    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_full,
        test_size=0.25,
        random_state=42,
        stratify=y_full,
    )
    return {
        'meta': meta,
        'data': level3_serialize_split(X_train, X_test, y_train, y_test, X_full, y_full),
        'model': None,
        'cell_runs': {
            'load_dataset': False,
            'define_model': False,
            'forward_pass': False,
            'train_model': False,
            'inspect': False,
            'evaluate': False,
        },
        'forward_summary': None,
        'training_logs': [],
        'evaluation': None,
        'inspect_ran': False,
    }


def level3_model_matches(store, meta):
    model = store.get('model')
    if model is None:
        return False

    model_meta = model.get('meta', {})
    return (
        model_meta.get('hidden_layer_sizes') == meta['hidden_layer_sizes']
        and model_meta.get('activation') == meta['activation']
        and model_meta.get('dataset') == meta['dataset']
    )


def level3_initialize_model(store, meta):
    X_train, _, y_train, _, _, _ = level3_deserialize_split(store['data'])
    model = init_level2_mlp(
        input_dim=2,
        hidden_layers=meta['hidden_layer_sizes'],
        output_dim=1,
    )
    model['meta'] = {
        'hidden_layers': meta['depth'],
        'neurons_per_layer': meta['width'],
        'hidden_layer_sizes': meta['hidden_layer_sizes'],
        'activation': meta['activation'],
        'dataset': meta['dataset'],
        'layer_sizes': meta['layer_sizes'],
    }
    model = level2_set_baseline_history(X_train, y_train, model, meta['activation'], l2=1e-4)
    store['model'] = model
    store['forward_summary'] = None
    store['evaluation'] = None
    store['inspect_ran'] = False
    store['cell_runs']['forward_pass'] = False
    store['cell_runs']['train_model'] = False
    store['cell_runs']['inspect'] = False
    store['cell_runs']['evaluate'] = False
    return store


def level3_dataset_preview_figure(X_train, X_test, y_train, y_test, dataset):
    fig = go.Figure()
    split_specs = [
        ('Train class 0', X_train[y_train == 0], '#0f766e', 'circle'),
        ('Train class 1', X_train[y_train == 1], '#b91c1c', 'circle'),
        ('Test class 0', X_test[y_test == 0], '#14b8a6', 'diamond-open'),
        ('Test class 1', X_test[y_test == 1], '#f97316', 'diamond-open'),
    ]

    for label, points, color, symbol in split_specs:
        if len(points) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode='markers',
            name=label,
            marker=dict(size=8, color=color, symbol=symbol, line=dict(width=1, color='white')),
        ))

    fig.update_layout(
        title=f"Dataset preview: {dataset.title()} split into train/test batches",
        xaxis_title='x1',
        yaxis_title='x2',
        margin=dict(l=30, r=10, t=50, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )
    return fig


def level3_activation_heatmap_figure(model, X_reference, activation):
    _, cache = level2_forward_pass(X_reference, model, activation)
    hidden_activations = cache['activations'][1:-1]
    if not hidden_activations:
        return make_level3_placeholder_figure('Per-layer activations', 'Define a model with at least one hidden layer.')

    fig = make_subplots(
        rows=len(hidden_activations),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[f'Hidden layer {index + 1}' for index in range(len(hidden_activations))]
    )

    for index, activation_matrix in enumerate(hidden_activations, start=1):
        z_values = activation_matrix[:min(12, activation_matrix.shape[0]), :min(80, activation_matrix.shape[1])]
        fig.add_trace(
            go.Heatmap(
                z=z_values,
                colorscale='RdBu',
                zmid=0,
                showscale=(index == 1),
                colorbar=dict(title='Activation') if index == 1 else None,
            ),
            row=index,
            col=1,
        )
        fig.update_yaxes(title_text='Neuron', row=index, col=1)

    fig.update_xaxes(title_text='Sample index', row=len(hidden_activations), col=1)
    fig.update_layout(
        title='Per-layer activations across a held-out batch',
        height=max(280, 220 * len(hidden_activations)),
        margin=dict(l=40, r=10, t=60, b=30),
    )
    return fig


def level3_hidden_space_figure(model, X_reference, y_reference, activation):
    _, cache = level2_forward_pass(X_reference, model, activation)
    hidden_activations = cache['activations'][1:-1]
    if not hidden_activations:
        return make_level3_placeholder_figure('Hidden-space projection', 'Run Cell 2 to define hidden layers.')

    last_hidden = hidden_activations[-1].T
    x_axis = last_hidden[:, 0]
    y_axis = last_hidden[:, 1] if last_hidden.shape[1] > 1 else np.zeros(last_hidden.shape[0])

    fig = go.Figure()
    for class_value, color in [(0, '#0f766e'), (1, '#b91c1c')]:
        mask = y_reference == class_value
        fig.add_trace(go.Scatter(
            x=x_axis[mask],
            y=y_axis[mask],
            mode='markers',
            name=f'Class {class_value}',
            marker=dict(size=9, color=color, line=dict(width=1, color='white')),
        ))

    fig.update_layout(
        title='Last hidden layer projection (neurons 1 and 2)',
        xaxis_title='Hidden dimension 1',
        yaxis_title='Hidden dimension 2',
        margin=dict(l=30, r=10, t=50, b=30),
    )
    return fig


def level3_confusion_matrix_figure(confusion_values):
    labels = ['Class 0', 'Class 1']
    fig = go.Figure(data=go.Heatmap(
        z=confusion_values,
        x=labels,
        y=labels,
        text=confusion_values,
        texttemplate='%{text}',
        textfont={'size': 14},
        colorscale='Blues',
        hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>',
    ))
    fig.update_layout(
        title='Confusion matrix on the evaluation split',
        xaxis_title='Predicted label',
        yaxis_title='True label',
        margin=dict(l=80, r=20, t=55, b=40),
    )
    return fig


def level3_misclassified_figure(X_test, y_test, pred_labels):
    misclassified = pred_labels != y_test
    fig = go.Figure()
    for class_value, color in [(0, '#94a3b8'), (1, '#475569')]:
        mask = y_test == class_value
        fig.add_trace(go.Scatter(
            x=X_test[mask, 0],
            y=X_test[mask, 1],
            mode='markers',
            name=f'Test class {class_value}',
            marker=dict(size=7, color=color),
        ))

    if np.any(misclassified):
        fig.add_trace(go.Scatter(
            x=X_test[misclassified, 0],
            y=X_test[misclassified, 1],
            mode='markers',
            name='Misclassified',
            marker=dict(size=11, color='#ef4444', symbol='x', line=dict(width=2)),
        ))

    fig.update_layout(
        title='Misclassified evaluation samples',
        xaxis_title='x1',
        yaxis_title='x2',
        margin=dict(l=30, r=10, t=50, b=30),
    )
    return fig


def level3_forward_summary_children(forward_summary):
    if not forward_summary:
        return html.Div('Run Cell 3 to inspect tensor shapes and example predictions.', style={'color': '#64748b'})

    example_rows = [
        html.Tr([
            html.Td(f"({example['x1']:.2f}, {example['x2']:.2f})"),
            html.Td(str(example['target'])),
            html.Td(f"{example['probability']:.3f}"),
        ])
        for example in forward_summary['examples']
    ]

    return html.Div([
        html.P(f"Batch shape: {tuple(forward_summary['batch_shape'])}"),
        html.P(f"Output shape: {tuple(forward_summary['output_shape'])}"),
        html.Ul([
            html.Li(f"Hidden layer {index + 1}: {tuple(shape)}")
            for index, shape in enumerate(forward_summary['hidden_shapes'])
        ], style={'fontFamily': 'monospace', 'fontSize': '12px'}),
        html.Table([
            html.Thead(html.Tr([html.Th('Input sample'), html.Th('Target'), html.Th('Predicted p(class=1)')])),
            html.Tbody(example_rows),
        ], style={'width': '100%', 'fontSize': '12px'})
    ])


def level3_training_log_children(training_logs):
    if not training_logs:
        return html.Div('Run Cell 4 to train the classifier and capture a notebook-style training log.', style={'color': '#64748b'})

    return html.Ul([
        html.Li(
            f"Run {entry['run_number']}: epochs={entry['epochs']}, train loss={entry['train_loss']:.4f}, "
            f"train acc={entry['train_accuracy'] * 100:.1f}%, test loss={entry['test_loss']:.4f}, "
            f"test acc={entry['test_accuracy'] * 100:.1f}%"
        )
        for entry in reversed(training_logs)
    ], style={'paddingLeft': '18px', 'fontSize': '12px'})


def level3_metrics_summary_children(evaluation):
    if not evaluation:
        return html.Div('Run Cell 6 to compute confusion, metrics, and misclassified samples.', style={'color': '#64748b'})

    precision = evaluation['precision']
    recall = evaluation['recall']
    f1 = evaluation['f1']
    support = evaluation['support']
    rows = []
    for index in range(len(precision)):
        rows.append(html.Tr([
            html.Td(f'Class {index}'),
            html.Td(f"{precision[index]:.2f}"),
            html.Td(f"{recall[index]:.2f}"),
            html.Td(f"{f1[index]:.2f}"),
            html.Td(str(support[index])),
        ]))

    return html.Div([
        html.P(f"Accuracy: {evaluation['metrics']['accuracy'] * 100:.1f}%"),
        html.P(f"Loss: {evaluation['metrics']['loss']:.4f}"),
        html.P(f"Misclassified points: {evaluation['misclassified_count']} / {evaluation['sample_count']}"),
        html.Table([
            html.Thead(html.Tr([
                html.Th('Class'), html.Th('Precision'), html.Th('Recall'), html.Th('F1'), html.Th('Support')
            ])),
            html.Tbody(rows)
        ], style={'width': '100%', 'fontSize': '12px'})
    ])


def level3_dataset_summary_children(X_train, X_test, y_train, y_test, meta):
    return html.Div([
        html.P(f"Dataset: {meta['dataset'].title()}"),
        html.P(f"Train split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples"),
        html.P(f"Class balance (train): class 0 = {int(np.sum(y_train == 0))}, class 1 = {int(np.sum(y_train == 1))}"),
        html.P(f"Feature space: 2-D input, {meta['depth']} hidden layer(s), {meta['width']} neuron(s) per hidden layer"),
    ], style={'fontSize': '12px'})


def level3_notebook_status_children(store):
    if store is None or not store['cell_runs']['load_dataset']:
        return html.Div('Start with Cell 1 to load a dataset and preview the classification split.')
    if not store['cell_runs']['define_model']:
        return html.Div('Dataset loaded. Next run Cell 2 to define the network before inspecting any activations.')
    if not store['cell_runs']['forward_pass']:
        return html.Div('Model defined. Cell 3 is the next useful step if you want to inspect tensor shapes before training.')
    if not store['cell_runs']['train_model']:
        return html.Div('Forward pass captured. Run Cell 4 to optimise the model and update the boundary.')
    if not store['cell_runs']['inspect']:
        return html.Div('Training complete. Run Cell 5 to examine hidden representations and per-layer activations.')
    if not store['cell_runs']['evaluate']:
        return html.Div('Inspection complete. Run Cell 6 to compute evaluation metrics and misclassified points.')
    return html.Div('All six notebook steps have been executed. Change a configuration and rerun the relevant cell to compare behaviours.')


@app.callback(
    Output('level3-params-store', 'data'),
    Input('level3-load-data-btn', 'n_clicks'),
    Input('level3-define-model-btn', 'n_clicks'),
    Input('level3-forward-btn', 'n_clicks'),
    Input('level3-train-btn', 'n_clicks'),
    Input('level3-inspect-btn', 'n_clicks'),
    Input('level3-evaluate-btn', 'n_clicks'),
    State('level3-dataset-dropdown', 'value'),
    State('level3-depth-slider', 'value'),
    State('level3-width-slider', 'value'),
    State('level3-activation-dropdown', 'value'),
    State('level3-epochs-slider', 'value'),
    State('level3-params-store', 'data'),
)
def update_level3_params(n_load, n_define, n_forward, n_train, n_inspect, n_evaluate,
                         dataset, depth, width, activation, epochs, store):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    meta = level3_build_meta(dataset, depth, width, activation, epochs)

    if store is None or store.get('meta', {}).get('dataset') != dataset:
        store = level3_initialize_store(meta)
    else:
        store['meta'] = meta

    if trigger in (None, 'level3-load-data-btn'):
        store = level3_initialize_store(meta)
        store['cell_runs']['load_dataset'] = True
        return store

    store['cell_runs']['load_dataset'] = True

    if trigger == 'level3-define-model-btn':
        store = level3_initialize_model(store, meta)
        store['cell_runs']['define_model'] = True
        return store

    if store.get('model') is None or not level3_model_matches(store, meta):
        store = level3_initialize_model(store, meta)
    store['cell_runs']['define_model'] = True

    X_train, X_test, y_train, y_test, _, _ = level3_deserialize_split(store['data'])

    if trigger == 'level3-forward-btn':
        batch_X = X_train[:24]
        batch_y = y_train[:24]
        predictions, cache = level2_forward_pass(batch_X, store['model'], meta['activation'])
        hidden_shapes = [list(activation_matrix.shape) for activation_matrix in cache['activations'][1:-1]]
        store['forward_summary'] = {
            'batch_shape': list(batch_X.shape),
            'hidden_shapes': hidden_shapes,
            'output_shape': list(predictions.shape),
            'examples': [
                {
                    'x1': batch_X[index, 0],
                    'x2': batch_X[index, 1],
                    'target': int(batch_y[index]),
                    'probability': float(predictions[index, 0]),
                }
                for index in range(min(5, batch_X.shape[0]))
            ],
        }
        store['cell_runs']['forward_pass'] = True

    elif trigger == 'level3-train-btn':
        store['model'] = train_level2_model(
            X_train,
            y_train,
            store['model'],
            activation=meta['activation'],
            epochs=meta['epochs'],
            lr=0.08,
            l2=1e-4,
        )
        train_metrics = level2_evaluate_metrics(X_train, y_train, store['model'], meta['activation'], l2=1e-4)
        test_metrics = level2_evaluate_metrics(X_test, y_test, store['model'], meta['activation'], l2=1e-4)
        store['training_logs'].append({
            'run_number': len(store['training_logs']) + 1,
            'epochs': meta['epochs'],
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
        })
        store['cell_runs']['train_model'] = True

    elif trigger == 'level3-inspect-btn':
        store['inspect_ran'] = True
        store['cell_runs']['inspect'] = True

    elif trigger == 'level3-evaluate-btn':
        predictions, _ = level2_forward_pass(X_test, store['model'], meta['activation'])
        pred_labels = (predictions.flatten() >= 0.5).astype(np.int32)
        confusion_values = confusion_matrix(y_test, pred_labels, labels=[0, 1])
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test,
            pred_labels,
            labels=[0, 1],
            zero_division=0,
        )
        metrics = level2_evaluate_metrics(X_test, y_test, store['model'], meta['activation'], l2=1e-4)
        store['evaluation'] = {
            'metrics': metrics,
            'confusion_matrix': confusion_values.tolist(),
            'pred_labels': pred_labels.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist(),
            'misclassified_count': int(np.sum(pred_labels != y_test)),
            'sample_count': int(y_test.shape[0]),
        }
        store['cell_runs']['evaluate'] = True

    return store


@app.callback(
    Output('level3-boundary-graph', 'figure'),
    Output('level3-loss-graph', 'figure'),
    Output('level3-activations-graph', 'figure'),
    Output('level3-dataset-preview-graph', 'figure'),
    Output('level3-dataset-summary', 'children'),
    Output('level3-network-diagram-graph', 'figure'),
    Output('level3-arch-summary', 'children'),
    Output('level3-forward-output', 'children'),
    Output('level3-training-log', 'children'),
    Output('level3-hidden-space-graph', 'figure'),
    Output('level3-confusion-matrix-graph', 'figure'),
    Output('level3-misclassified-graph', 'figure'),
    Output('level3-metrics-summary', 'children'),
    Output('level3-notebook-status', 'children'),
    Input('level3-params-store', 'data'),
)
def update_level3_views(store):
    boundary_placeholder = make_level3_placeholder_figure(
        'Decision boundary / prediction surface',
        'Run Cell 2 to define a classifier and render its decision surface.'
    )
    loss_placeholder = make_level3_placeholder_figure(
        'Loss curve',
        'Run Cell 4 to train the model and record optimisation history.'
    )
    activation_placeholder = make_level3_placeholder_figure(
        'Per-layer activations',
        'Run Cell 5 to inspect hidden activations after defining the model.'
    )
    dataset_placeholder = make_level3_placeholder_figure(
        'Dataset preview',
        'Run Cell 1 to load a toy classification dataset.'
    )
    model_placeholder = make_level3_placeholder_figure(
        'Architecture diagram',
        'Run Cell 2 to define the hidden stack and output layer.'
    )
    hidden_placeholder = make_level3_placeholder_figure(
        'Hidden-space projection',
        'Run Cell 5 to inspect hidden representations.'
    )
    eval_placeholder = make_level3_placeholder_figure(
        'Evaluation',
        'Run Cell 6 to compute confusion and evaluation diagnostics.'
    )

    if store is None:
        return (
            boundary_placeholder,
            loss_placeholder,
            activation_placeholder,
            dataset_placeholder,
            html.Div('Run Cell 1 to create the dataset preview.', style={'color': '#64748b'}),
            model_placeholder,
            html.Div('Run Cell 2 to generate the architecture summary.', style={'color': '#64748b'}),
            level3_forward_summary_children(None),
            level3_training_log_children([]),
            hidden_placeholder,
            eval_placeholder,
            eval_placeholder,
            level3_metrics_summary_children(None),
            level3_notebook_status_children(None),
        )

    meta = store['meta']
    X_train, X_test, y_train, y_test, X_full, y_full = level3_deserialize_split(store['data'])
    dataset_preview = level3_dataset_preview_figure(X_train, X_test, y_train, y_test, meta['dataset'])
    dataset_summary = level3_dataset_summary_children(X_train, X_test, y_train, y_test, meta)
    notebook_status = level3_notebook_status_children(store)

    boundary_fig = boundary_placeholder
    loss_fig = loss_placeholder
    activations_fig = activation_placeholder
    network_fig = model_placeholder
    arch_summary = html.Div('Run Cell 2 to generate the architecture summary.', style={'color': '#64748b'})
    hidden_space_fig = hidden_placeholder
    confusion_fig = eval_placeholder
    misclassified_fig = eval_placeholder

    model = store.get('model')
    if model is not None:
        boundary_fig = make_decision_boundary_figure(X_full, y_full, model, meta['activation'])
        if store['cell_runs']['train_model']:
            boundary_fig.update_layout(title=f"Trained decision boundary after {model.get('epoch', 0)} epochs")
        else:
            boundary_fig.update_layout(title='Initial decision surface from the current model definition')

        history = model.get('history', {})
        if history.get('loss'):
            loss_fig = make_level2_training_curves_figure(history)
            loss_fig.update_layout(title='Training loss and accuracy from Cell 4')

        network_fig = make_network_diagram_figure(
            input_dim=2,
            hidden_layers=meta['hidden_layer_sizes'],
            output_dim=1,
            params=model,
            activation=meta['activation'],
        )
        arch_summary = make_level2_summary_panel(model, meta['activation'])

        if store['inspect_ran']:
            activations_fig = level3_activation_heatmap_figure(model, X_test, meta['activation'])
            hidden_space_fig = level3_hidden_space_figure(model, X_test, y_test, meta['activation'])

    forward_output = level3_forward_summary_children(store.get('forward_summary'))
    training_log = level3_training_log_children(store.get('training_logs', []))

    evaluation = store.get('evaluation')
    if evaluation is not None:
        confusion_fig = level3_confusion_matrix_figure(evaluation['confusion_matrix'])
        pred_labels = np.array(evaluation['pred_labels'], dtype=np.int32)
        misclassified_fig = level3_misclassified_figure(X_test, y_test, pred_labels)
    metrics_summary = level3_metrics_summary_children(evaluation)

    return (
        boundary_fig,
        loss_fig,
        activations_fig,
        dataset_preview,
        dataset_summary,
        network_fig,
        arch_summary,
        forward_output,
        training_log,
        hidden_space_fig,
        confusion_fig,
        misclassified_fig,
        metrics_summary,
        notebook_status,
    )

#endregion

if __name__ == '__main__':
    app.run(debug=False)   #changed debug to false because otherwise it resets the page every minute


