#region Imports
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
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
from pagelayout import level4_layout
from pagelayout import level5_layout

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

def init_single_hidden_mlp(input_dim=2, hidden_dim=4, output_dim=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Xavier / Glorot-like scaling for stability
    W1 = rng.normal(0.0, 1.0 / np.sqrt(input_dim), size=(hidden_dim, input_dim))
    b1 = np.zeros((hidden_dim, 1))
    W2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), size=(output_dim, hidden_dim))
    b2 = np.zeros((output_dim, 1))

    return {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
    }

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

def forward_pass(X, params, activation):
    """
    X: (N, 2)
    returns dict with intermediate values for backprop
    """
    W1, b1 = params['W1'], params['b1']  # (H,2), (H,1)
    W2, b2 = params['W2'], params['b2']  # (1,H), (1,1)

    X_t = X.T  # (2, N)

    Z1 = W1 @ X_t + b1          # (H, N)
    A1 = activation_forward(Z1, activation)  # (H, N)

    Z2 = W2 @ A1 + b2           # (1, N)
    A2 = 1.0 / (1.0 + np.exp(-Z2))  # sigmoid output (1, N)

    cache = {
        'X_t': X_t,
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2,
    }
    return A2, cache

def train_single_hidden_step(X, y, params, activation='tanh',
                             steps=50, lr=0.1, l2=0.0):
    """
    Run a few GD steps to make the boundary visibly change.
    X: (N, 2), y: (N,)
    """
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    y_row = y.reshape(1, -1)  # (1, N)

    for _ in range(steps):
        # Forward
        A2, cache = forward_pass(X, params, activation)
        A1, X_t = cache['A1'], cache['X_t']

        # Binary cross-entropy derivative wrt A2
        eps = 1e-8
        dA2 = -(y_row / (A2 + eps) - (1 - y_row) / (1 - A2 + eps))  # (1, N)

        # dZ2 = dL/dA2 * dA2/dZ2
        dZ2 = dA2 * A2 * (1 - A2)  # sigmoid prime

        # Gradients for W2, b2
        dW2 = (dZ2 @ A1.T) / X.shape[0]  # (1, H)
        db2 = np.mean(dZ2, axis=1, keepdims=True)  # (1,1)

        # Backprop into hidden layer
        dA1 = W2.T @ dZ2  # (H, N)
        dZ1 = activation_backward(dA1, cache['Z1'], activation)  # (H, N)

        dW1 = (dZ1 @ X_t.T) / X.shape[0]  # (H, 2)
        db1 = np.mean(dZ1, axis=1, keepdims=True)  # (H,1)

        # Optional L2 regularization
        if l2 > 0:
            dW2 += l2 * W2
            dW1 += l2 * W1

        # Gradient descent update
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

        params.update({'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2})

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
    A2_grid, _ = forward_pass(grid_points, params, activation)
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

def make_network_diagram_figure(input_dim=2, hidden_dim=4, output_dim=1,
                                params=None, activation='tanh'):
    # x positions for 3 layers
    x_in, x_hid, x_out = 0, 1, 2

    # y positions: spread nodes vertically
    y_in = np.linspace(0, 1, input_dim)
    y_hid = np.linspace(0, 1, hidden_dim)
    y_out = np.linspace(0, 1, output_dim)

    # Extract weight matrices if available
    W1 = np.array(params['W1']) if params else None  # (H, 2)
    b1 = np.array(params['b1']) if params else None  # (H, 1)
    W2 = np.array(params['W2']) if params else None  # (1, H)
    b2 = np.array(params['b2']) if params else None  # (1, 1)

    nodes_x = []
    nodes_y = []
    text = []
    hover_text = []
    layer_colors = []

    # Input nodes
    for i in range(input_dim):
        nodes_x.append(x_in)
        nodes_y.append(y_in[i])
        text.append(f"x{i+1}")
        hover_text.append(f"Input x{i+1}")
        layer_colors.append("lightblue")

    # Hidden nodes - show neuron equation on hover
    for j in range(hidden_dim):
        nodes_x.append(x_hid)
        nodes_y.append(y_hid[j])
        text.append(f"h{j+1}")
        if W1 is not None and b1 is not None:
            w_str = ' + '.join(f'{W1[j, i]:+.2f}·x{i+1}' for i in range(input_dim))
            b_val = b1[j, 0]
            hover_text.append(
                f"z{j+1} = {w_str} {b_val:+.2f}<br>"
                f"a{j+1} = {activation}(z{j+1})"
            )
        else:
            hover_text.append(f"h{j+1} = {activation}(w⊤x + b)")
        layer_colors.append("lightgreen")

    # Output nodes
    for k in range(output_dim):
        nodes_x.append(x_out)
        nodes_y.append(y_out[k])
        text.append(f"ŷ")
        if W2 is not None and b2 is not None:
            w_str = ' + '.join(f'{W2[k, j]:+.2f}·a{j+1}' for j in range(hidden_dim))
            b_val = b2[k, 0]
            hover_text.append(
                f"z_out = {w_str} {b_val:+.2f}<br>"
                f"ŷ = σ(z_out)"
            )
        else:
            hover_text.append("ŷ = σ(w⊤a + b)")
        layer_colors.append("salmon")

    node_trace = go.Scatter(
        x=nodes_x,
        y=nodes_y,
        mode='markers+text',
        text=text,
        textposition='middle right',
        hovertext=hover_text,
        hoverinfo='text',
        marker=dict(size=18, color=layer_colors, line=dict(width=1, color='black'))
    )

    # Edges as separate Scatter traces (lines) with hover showing weight value
    edge_traces = []

    # Input -> Hidden edges
    for i in range(input_dim):
        for j in range(hidden_dim):
            if W1 is not None:
                w_val = W1[j, i]
                edge_hover = f"w¹[{j+1},{i+1}] = {w_val:.3f}<br>x{i+1} → h{j+1}"
                edge_width = max(0.5, min(4, abs(w_val) * 3))
                edge_color = 'steelblue' if w_val >= 0 else 'crimson'
            else:
                edge_hover = f"w¹[{j+1},{i+1}]"
                edge_width = 1
                edge_color = 'grey'
            # Midpoint for hover target
            mx = (x_in + x_hid) / 2
            my = (y_in[i] + y_hid[j]) / 2
            edge_traces.append(go.Scatter(
                x=[x_in, mx, x_hid],
                y=[y_in[i], my, y_hid[j]],
                mode='lines',
                line=dict(color=edge_color, width=edge_width),
                hovertext=[None, edge_hover, None],
                hoverinfo='text',
                showlegend=False
            ))

    # Hidden -> Output edges
    for j in range(hidden_dim):
        for k in range(output_dim):
            if W2 is not None:
                w_val = W2[k, j]
                edge_hover = f"w²[{k+1},{j+1}] = {w_val:.3f}<br>h{j+1} → ŷ"
                edge_width = max(0.5, min(4, abs(w_val) * 3))
                edge_color = 'steelblue' if w_val >= 0 else 'crimson'
            else:
                edge_hover = f"w²[{k+1},{j+1}]"
                edge_width = 1
                edge_color = 'grey'
            mx = (x_hid + x_out) / 2
            my = (y_hid[j] + y_out[k]) / 2
            edge_traces.append(go.Scatter(
                x=[x_hid, mx, x_out],
                y=[y_hid[j], my, y_out[k]],
                mode='lines',
                line=dict(color=edge_color, width=edge_width),
                hovertext=[None, edge_hover, None],
                hoverinfo='text',
                showlegend=False
            ))

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Single hidden layer network (hover for w⊤x + b)",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    return fig

@app.callback(
    Output('level2-params-store', 'data'),
    Input('level2-randomize-btn', 'n_clicks'),
    Input('level2-trainstep-btn', 'n_clicks'),
    State('level2-width-slider', 'value'),
    State('level2-activation-dropdown', 'value'),
    State('level2-dataset-dropdown', 'value'),
    State('level2-params-store', 'data'),
)
def update_level2_params(n_rand, n_step, width, activation, dataset, params):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # 1) On first use or randomize: initialize weights/biases for 2D->width->1 MLP
    if params is None or trigger == 'level2-randomize-btn':
        params = init_single_hidden_mlp(input_dim=2, hidden_dim=width, output_dim=1)

    # 2) On train step: run a few gradient steps on the chosen toy dataset
    if trigger == 'level2-trainstep-btn':
        X, y = load_toy_dataset(dataset)  # e.g. moons/circles/linear
        params = train_single_hidden_step(X, y, params, activation=activation, steps=50)

    # Also store meta: width, activation, dataset for the view callback
    params['meta'] = {'width': width, 'activation': activation, 'dataset': dataset}
    return params

@app.callback(
    Output('level2-decision-boundary-graph', 'figure'),
    Output('level2-activation-graph', 'figure'),
    Output('level2-network-diagram-graph', 'figure'),
    Output('level2-math-explanation', 'children'),
    Input('level2-params-store', 'data')
)
def update_level2_views(params):
    if params is None:
        raise dash.exceptions.PreventUpdate

    meta = params.get('meta', {})
    width = meta.get('width', 4)
    activation = meta.get('activation', 'tanh')
    dataset = meta.get('dataset', 'moons')

    # 1) Decision boundary figure
    X, y = load_toy_dataset(dataset)          # same helper as above
    fig_boundary = make_decision_boundary_figure(X, y, params, activation)
    # Update title to highlight linear→curve bending
    if dataset == 'linear':
        fig_boundary.update_layout(title="Decision boundary (hidden layer can separate linear data)")
    else:
        fig_boundary.update_layout(title=f"Decision boundary on '{dataset}' (hidden layer bends line → curves)")

    # 2) Activation function figure
    fig_activation = make_activation_figure(activation)

    # 3) Network diagram with actual weights for hover
    fig_network = make_network_diagram_figure(
        input_dim=2, hidden_dim=width, output_dim=1,
        params=params, activation=activation
    )

    # 4) Math explanation – neuron equation, per-neuron detail, parameter counting
    n_in, n_h, n_out = 2, width, 1
    num_params_W1 = n_in * n_h
    num_params_b1 = n_h
    num_params_W2 = n_h * n_out
    num_params_b2 = n_out
    total_params = num_params_W1 + num_params_b1 + num_params_W2 + num_params_b2

    W1 = np.array(params['W1'])  # (H, 2)
    b1 = np.array(params['b1'])  # (H, 1)
    W2 = np.array(params['W2'])  # (1, H)
    b2 = np.array(params['b2'])  # (1, 1)

    # Per-neuron equations for the hidden layer
    neuron_items = []
    for j in range(n_h):
        w_terms = ' + '.join(f'{W1[j, i]:+.2f}·x{i+1}' for i in range(n_in))
        b_val = b1[j, 0]
        neuron_items.append(
            html.Li(f"h{j+1}: z = {w_terms} {b_val:+.2f},  a = {activation}(z)")
        )

    explanation = html.Div([
        html.H5("Neuron equation"),
        html.P("a = ρ(w⊤x + b)"),
        html.P(f"where ρ = {activation}, composing a linear map (w⊤x + b) with a nonlinearity."),
        html.Hr(),
        html.H5("Hidden layer neurons"),
        html.Ul(neuron_items, style={'fontSize': '11px', 'fontFamily': 'monospace'}),
        html.Hr(),
        html.H5("Parameter count"),
        html.Ul([
            html.Li(f"W¹: {n_h}×{n_in} = {num_params_W1} weights"),
            html.Li(f"b¹: {n_h} biases"),
            html.Li(f"W²: {n_out}×{n_h} = {num_params_W2} weights"),
            html.Li(f"b²: {n_out} biases"),
            html.Li(html.B(f"Total: {total_params} trainable parameters")),
        ]),
    ])

    return fig_boundary, fig_activation, fig_network, explanation

#endregion

#====DECLARATION: CODE HERE IS ASSISTED BY Copilot (GPT-5.4) 22/03/26 - 23/04/26====
#region MODEL FACTORY: Level 3 callbacks/methods to illustrate a deeper network structure (why is depth helpful) - WORK IN PROGRESS

def level3_target_function(x, name):
    """Return y values for the chosen 1-D target function."""
    if name == 'sin':
        return np.sin(x)
    elif name == 'abs':
        return np.abs(x)
    elif name == 'quadratic':
        return x ** 2
    elif name == 'step':
        return (x >= 0).astype(float)
    elif name == 'sawtooth':
        return x - np.floor(x)
    return np.sin(x)


def init_deep_mlp(input_dim, hidden_width, depth, output_dim=1):
    """Initialise a variable-depth MLP with Xavier scaling."""
    rng = np.random.default_rng()
    params = {'depth': depth, 'width': hidden_width}

    # First hidden layer
    fan_in = input_dim
    params['W0'] = rng.normal(0, 1.0 / np.sqrt(fan_in),
                              size=(hidden_width, fan_in)).tolist()
    params['b0'] = np.zeros(hidden_width).tolist()

    # Additional hidden layers
    for d in range(1, depth):
        fan_in = hidden_width
        params[f'W{d}'] = rng.normal(0, 1.0 / np.sqrt(fan_in),
                                     size=(hidden_width, fan_in)).tolist()
        params[f'b{d}'] = np.zeros(hidden_width).tolist()

    # Output layer
    fan_in = hidden_width
    params[f'W{depth}'] = rng.normal(0, 1.0 / np.sqrt(fan_in),
                                     size=(output_dim, fan_in)).tolist()
    params[f'b{depth}'] = np.zeros(output_dim).tolist()

    params['loss_history'] = []
    return params


def deep_mlp_forward(x_col, params, activation, return_intermediates=False):
    """
    Forward pass through a variable-depth MLP.
    x_col: (N, 1)  returns (N, 1) output and optionally layer activations.
    """
    depth = params['depth']
    A = x_col.T  # (1, N)

    intermediates = []

    for d in range(depth):
        W = np.array(params[f'W{d}'])
        b = np.array(params[f'b{d}']).reshape(-1, 1)
        Z = W @ A + b
        A = activation_forward(Z, activation)
        if return_intermediates:
            intermediates.append(A.copy())

    # Output layer (linear)
    W_out = np.array(params[f'W{depth}'])
    b_out = np.array(params[f'b{depth}']).reshape(-1, 1)
    out = W_out @ A + b_out  # (1, N)

    if return_intermediates:
        return out.T, intermediates      # (N, 1), list of (width, N)
    return out.T                          # (N, 1)


def train_deep_mlp(x_train, y_train, params, activation, epochs=200, lr=0.01):
    """Train the deep MLP with simple GD on MSE loss.  Returns updated params."""
    depth = params['depth']
    N = x_train.shape[0]
    loss_history = list(params.get('loss_history', []))

    for _ in range(epochs):
        # --- forward ---
        As = [x_train.T]          # A[0] = input (1, N)
        Zs = []

        A = As[0]
        for d in range(depth):
            W = np.array(params[f'W{d}'])
            b = np.array(params[f'b{d}']).reshape(-1, 1)
            Z = W @ A + b
            Zs.append(Z)
            A = activation_forward(Z, activation)
            As.append(A)

        W_out = np.array(params[f'W{depth}'])
        b_out = np.array(params[f'b{depth}']).reshape(-1, 1)
        Z_out = W_out @ A + b_out          # (1, N)
        y_pred = Z_out                     # linear output

        # MSE loss
        diff = y_pred - y_train.T           # (1, N)
        loss = float(np.mean(diff ** 2))
        loss_history.append(loss)

        # --- backward ---
        dZ = 2.0 * diff / N                # (1, N)

        # Output layer grads
        dW_out = dZ @ As[depth].T           # (1, width)
        db_out = np.sum(dZ, axis=1, keepdims=True)

        dA = W_out.T @ dZ                  # (width, N)

        # Update output layer
        W_out -= lr * dW_out
        b_out -= lr * db_out
        params[f'W{depth}'] = W_out.tolist()
        params[f'b{depth}'] = b_out.flatten().tolist()

        # Hidden layers (reverse)
        for d in reversed(range(depth)):
            dZ_h = activation_backward(dA, Zs[d], activation)
            W = np.array(params[f'W{d}'])
            b = np.array(params[f'b{d}']).reshape(-1, 1)

            dW = dZ_h @ As[d].T
            db = np.sum(dZ_h, axis=1, keepdims=True)

            dA = W.T @ dZ_h

            W -= lr * dW
            b -= lr * db
            params[f'W{d}'] = W.tolist()
            params[f'b{d}'] = b.flatten().tolist()

    params['loss_history'] = loss_history
    return params

@app.callback(
    Output('level3-params-store', 'data'),
    Input('level3-randomize-btn', 'n_clicks'),
    Input('level3-train-btn', 'n_clicks'),
    State('level3-width-slider', 'value'),
    State('level3-depth-slider', 'value'),
    State('level3-activation-dropdown', 'value'),
    State('level3-target-dropdown', 'value'),
    State('level3-epochs-slider', 'value'),
    State('level3-params-store', 'data'),
)
def update_level3_params(n_rand, n_train, width, depth, activation,
                         target, epochs, params):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Initialise on first visit or explicit randomize
    if params is None or trigger == 'level3-randomize-btn':
        params = init_deep_mlp(input_dim=1, hidden_width=width, depth=depth)

    # Train
    if trigger == 'level3-train-btn':
        x_train = np.linspace(-2 * np.pi, 2 * np.pi, 400).reshape(-1, 1)
        y_train = level3_target_function(x_train.flatten(), target).reshape(-1, 1)
        params = train_deep_mlp(x_train, y_train, params, activation,
                                epochs=epochs, lr=0.005)

    # Store meta for the view callback
    params['meta'] = {
        'width': width, 'depth': depth,
        'activation': activation, 'target': target,
    }
    return params


@app.callback(
    Output('level3-approx-graph', 'figure'),
    Output('level3-loss-graph', 'figure'),
    Output('level3-activations-graph', 'figure'),
    Output('level3-arch-summary', 'children'),
    Input('level3-params-store', 'data'),
)
def update_level3_views(params):
    if params is None:
        raise dash.exceptions.PreventUpdate

    meta = params.get('meta', {})
    width = meta.get('width', 8)
    depth = meta.get('depth', 1)
    activation = meta.get('activation', 'relu')
    target = meta.get('target', 'sin')

    x_plot = np.linspace(-2 * np.pi, 2 * np.pi, 500).reshape(-1, 1)
    y_target = level3_target_function(x_plot.flatten(), target)

    # Network prediction + intermediate activations
    y_pred, intermediates = deep_mlp_forward(x_plot, params, activation,
                                             return_intermediates=True)
    y_pred = y_pred.flatten()

    # 1) Side-by-side target vs approximation
    fig_approx = go.Figure()
    fig_approx.add_trace(go.Scatter(
        x=x_plot.flatten(), y=y_target, mode='lines',
        name='Target f(x)', line=dict(color='black', width=2, dash='dash')
    ))
    fig_approx.add_trace(go.Scatter(
        x=x_plot.flatten(), y=y_pred, mode='lines',
        name='Network output', line=dict(color='crimson', width=2)
    ))
    fig_approx.update_layout(
        title=f"Universal approximation: {target}(x) vs network "
              f"(depth={depth}, width={width}, act={activation})",
        xaxis_title="x", yaxis_title="y",
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=40, r=10, t=40, b=30),
    )

    # 2) Loss curve
    loss_history = params.get('loss_history', [])
    fig_loss = go.Figure()
    if loss_history:
        fig_loss.add_trace(go.Scatter(
            y=loss_history, mode='lines',
            name='MSE Loss', line=dict(color='steelblue')
        ))
    fig_loss.update_layout(
        title="Training loss (MSE)",
        xaxis_title="Epoch", yaxis_title="Loss",
        margin=dict(l=40, r=10, t=40, b=30),
    )

    # 3) Per-layer activations plot (show first 5 neurons of each layer)
    fig_acts = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for layer_idx, act_matrix in enumerate(intermediates):
        n_show = min(5, act_matrix.shape[0])
        for neuron in range(n_show):
            fig_acts.add_trace(go.Scatter(
                x=x_plot.flatten(),
                y=act_matrix[neuron, :],
                mode='lines',
                name=f'L{layer_idx+1} n{neuron+1}',
                line=dict(width=1, color=colors[layer_idx % len(colors)]),
                opacity=0.7,
                showlegend=(neuron == 0),
                legendgroup=f'layer{layer_idx}',
            ))
    fig_acts.update_layout(
        title="Hidden neuron activations per layer",
        xaxis_title="x", yaxis_title="activation",
        margin=dict(l=40, r=10, t=40, b=30),
    )

    # 4) Architecture summary
    total_params = 0
    layer_info = []
    prev_dim = 1  # input dim
    for d in range(depth):
        n_w = prev_dim * width
        n_b = width
        total_params += n_w + n_b
        layer_info.append(
            html.Li(f"Hidden {d+1}: {prev_dim} → {width} "
                     f"({n_w} weights + {n_b} biases)")
        )
        prev_dim = width
    # Output layer
    n_w = prev_dim * 1
    n_b = 1
    total_params += n_w + n_b
    layer_info.append(
        html.Li(f"Output: {prev_dim} → 1 ({n_w} weights + {n_b} bias)")
    )

    summary = html.Div([
        html.P(f"Input: 1 feature (x)"),
        html.P(f"Architecture: {depth} hidden layer{'s' if depth > 1 else ''}, "
               f"{width} neurons each, activation = {activation}"),
        html.Ul(layer_info, style={'fontSize': '12px', 'fontFamily': 'monospace'}),
        html.P(html.B(f"Total trainable parameters: {total_params}")),
        html.Hr(),
        html.P("Insight: as width or depth increases, the network can represent "
               "increasingly complex functions — this is the universal approximation "
               "theorem in action. ReLU networks build piecewise-linear fits; "
               "tanh/sigmoid can produce smoother curves.",
               style={'fontSize': '12px', 'fontStyle': 'italic'}),
    ])

    return fig_approx, fig_loss, fig_acts, summary

#endregion

#====DECLARATION: CODE HERE IS ASSISTED BY Copilot (GPT-5.4) 22/03/26 - 23/04/26====
#region MODEL FACTORY: Level 4 callbacks/methods to show trainin dynamics of FNN (forward pass/ backprop pass etc) - WORK IN PROGRESS
def l4_init_mlp(hidden_dim, activation, dataset, val_split):
    """Initialise a 2→H→1 MLP and split toy data into train/val."""
    X_all, y_all = load_toy_dataset(dataset)
    N = X_all.shape[0]
    n_val = max(1, int(N * val_split))
    # Fixed shuffle so data stays consistent within a session
    rng = np.random.default_rng(0)
    idx = rng.permutation(N)
    X_all, y_all = X_all[idx], y_all[idx]

    X_val, y_val = X_all[:n_val], y_all[:n_val]
    X_train, y_train = X_all[n_val:], y_all[n_val:]

    params = init_single_hidden_mlp(input_dim=2, hidden_dim=hidden_dim, output_dim=1)
    # Convert numpy arrays to lists for JSON serialisation
    for k in ['W1', 'b1', 'W2', 'b2']:
        params[k] = np.array(params[k]).tolist()

    params['train_loss'] = []
    params['val_loss'] = []
    params['train_acc'] = []
    params['val_acc'] = []
    params['epoch'] = 0
    params['grad_norms'] = []  # per-layer gradient norms for the last step
    params['X_train'] = X_train.tolist()
    params['y_train'] = y_train.tolist()
    params['X_val'] = X_val.tolist()
    params['y_val'] = y_val.tolist()
    return params


def l4_compute_loss_acc(X, y, params, activation):
    """Binary cross-entropy loss and accuracy for 2→H→1 MLP."""
    for k in ['W1', 'b1', 'W2', 'b2']:
        params[k] = np.array(params[k])
    A2, _ = forward_pass(X, params, activation)
    preds = (A2.flatten() >= 0.5).astype(int)
    acc = float(np.mean(preds == y))
    eps = 1e-8
    y_row = y.reshape(1, -1).astype(float)
    loss = -float(np.mean(y_row * np.log(A2 + eps) + (1 - y_row) * np.log(1 - A2 + eps)))
    for k in ['W1', 'b1', 'W2', 'b2']:
        params[k] = np.array(params[k]).tolist()
    return loss, acc


def l4_train_step(params, activation, lr, n_steps=1):
    """Run n_steps of gradient descent, recording loss/acc/grad norms."""
    X_train = np.array(params['X_train'])
    y_train = np.array(params['y_train'])
    X_val = np.array(params['X_val'])
    y_val = np.array(params['y_val'])

    # Restore numpy arrays
    for k in ['W1', 'b1', 'W2', 'b2']:
        params[k] = np.array(params[k])

    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    y_row = y_train.reshape(1, -1).astype(float)
    N = X_train.shape[0]

    grad_norms_last = []

    for _ in range(n_steps):
        # Forward
        A2, cache = forward_pass(X_train, params, activation)
        A1 = cache['A1']
        X_t = cache['X_t']

        # BCE gradient
        eps = 1e-8
        dA2 = -(y_row / (A2 + eps) - (1 - y_row) / (1 - A2 + eps))
        dZ2 = dA2 * A2 * (1 - A2)

        dW2 = (dZ2 @ A1.T) / N
        db2 = np.mean(dZ2, axis=1, keepdims=True)

        dA1 = W2.T @ dZ2
        dZ1 = activation_backward(dA1, cache['Z1'], activation)
        dW1 = (dZ1 @ X_t.T) / N
        db1 = np.mean(dZ1, axis=1, keepdims=True)

        # Record gradient norms
        grad_norms_last = [
            float(np.linalg.norm(dW1)),
            float(np.linalg.norm(db1)),
            float(np.linalg.norm(dW2)),
            float(np.linalg.norm(db2)),
        ]

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
        params.update({'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2})

        # Metrics
        t_loss, t_acc = l4_compute_loss_acc(X_train, y_train, params, activation)
        v_loss, v_acc = l4_compute_loss_acc(X_val, y_val, params, activation)
        params['train_loss'].append(t_loss)
        params['val_loss'].append(v_loss)
        params['train_acc'].append(t_acc)
        params['val_acc'].append(v_acc)
        params['epoch'] += 1

    params['grad_norms'] = grad_norms_last

    # Serialise back to lists
    for k in ['W1', 'b1', 'W2', 'b2']:
        params[k] = np.array(params[k]).tolist()
    return params


def l4_make_gradient_flow_figure(params, hidden_dim, activation):
    """Diagram showing gradient magnitudes flowing backward through layers."""
    grad_norms = params.get('grad_norms', [])
    if not grad_norms:
        grad_norms = [0, 0, 0, 0]

    layer_names = ['dW¹ (input→hidden)', 'db¹ (hidden bias)',
                   'dW² (hidden→output)', 'db² (output bias)']
    # Reverse so output is on top (backprop direction)
    layer_names_rev = layer_names[::-1]
    grad_norms_rev = grad_norms[::-1]

    colors = ['#ef4444', '#f97316', '#3b82f6', '#22c55e']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grad_norms_rev,
        y=layer_names_rev,
        orientation='h',
        marker_color=colors,
        text=[f'{g:.4f}' for g in grad_norms_rev],
        textposition='outside',
    ))
    fig.update_layout(
        title='Gradient magnitude (backward flow ← output to input)',
        xaxis_title='||gradient||',
        margin=dict(l=10, r=10, t=40, b=30),
        height=250,
    )
    return fig


# --- Level 4 callbacks ---

@app.callback(
    Output('level4-params-store', 'data'),
    Input('level4-reset-btn', 'n_clicks'),
    Input('level4-step-btn', 'n_clicks'),
    Input('level4-train50-btn', 'n_clicks'),
    State('level4-width-slider', 'value'),
    State('level4-activation-dropdown', 'value'),
    State('level4-dataset-dropdown', 'value'),
    State('level4-lr-slider', 'value'),
    State('level4-split-slider', 'value'),
    State('level4-params-store', 'data'),
)
def update_level4_params(n_reset, n_step, n_train50,
                         width, activation, dataset, log_lr, val_split, params):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    lr = 10 ** log_lr

    if params is None or trigger == 'level4-reset-btn':
        params = l4_init_mlp(width, activation, dataset, val_split)

    if trigger == 'level4-step-btn':
        params = l4_train_step(params, activation, lr, n_steps=1)

    if trigger == 'level4-train50-btn':
        params = l4_train_step(params, activation, lr, n_steps=50)

    params['meta'] = {
        'width': width, 'activation': activation, 'dataset': dataset,
        'lr': lr, 'val_split': val_split,
    }
    return params


@app.callback(
    Output('level4-boundary-graph', 'figure'),
    Output('level4-loss-graph', 'figure'),
    Output('level4-accuracy-graph', 'figure'),
    Output('level4-gradient-graph', 'figure'),
    Output('level4-pass-explanation', 'children'),
    Output('level4-fit-status', 'children'),
    Output('level4-epoch-counter', 'children'),
    Input('level4-params-store', 'data'),
)
def update_level4_views(params):
    if params is None:
        raise dash.exceptions.PreventUpdate

    meta = params.get('meta', {})
    width = meta.get('width', 6)
    activation = meta.get('activation', 'tanh')
    dataset = meta.get('dataset', 'moons')
    lr = meta.get('lr', 0.03)
    epoch = params.get('epoch', 0)

    # Reconstruct numpy params for figures
    np_params = dict(params)
    for k in ['W1', 'b1', 'W2', 'b2']:
        np_params[k] = np.array(params[k])

    # 1) Decision boundary (train data)
    X_train = np.array(params['X_train'])
    y_train = np.array(params['y_train'])
    fig_boundary = make_decision_boundary_figure(X_train, y_train, np_params, activation)
    fig_boundary.update_layout(
        title=f"Decision boundary (epoch {epoch}, lr={lr:.4f})"
    )

    # 2) Loss curves
    fig_loss = go.Figure()
    if params['train_loss']:
        fig_loss.add_trace(go.Scatter(
            y=params['train_loss'], mode='lines',
            name='Train loss', line=dict(color='steelblue')
        ))
        fig_loss.add_trace(go.Scatter(
            y=params['val_loss'], mode='lines',
            name='Val loss', line=dict(color='crimson', dash='dash')
        ))
    fig_loss.update_layout(
        title='Loss over epochs',
        xaxis_title='Epoch', yaxis_title='BCE Loss',
        margin=dict(l=40, r=10, t=40, b=30),
    )

    # 3) Accuracy curves
    fig_acc = go.Figure()
    if params['train_acc']:
        fig_acc.add_trace(go.Scatter(
            y=params['train_acc'], mode='lines',
            name='Train acc', line=dict(color='steelblue')
        ))
        fig_acc.add_trace(go.Scatter(
            y=params['val_acc'], mode='lines',
            name='Val acc', line=dict(color='crimson', dash='dash')
        ))
    fig_acc.update_layout(
        title='Accuracy over epochs',
        xaxis_title='Epoch', yaxis_title='Accuracy',
        yaxis=dict(range=[0, 1.05]),
        margin=dict(l=40, r=10, t=40, b=30),
    )

    # 4) Gradient flow diagram
    fig_grad = l4_make_gradient_flow_figure(params, width, activation)

    # 5) Forward / backward pass walkthrough
    W1 = np.array(params['W1'])
    b1 = np.array(params['b1'])
    W2 = np.array(params['W2'])
    b2 = np.array(params['b2'])
    grad_norms = params.get('grad_norms', [0, 0, 0, 0])
    if not grad_norms:
        grad_norms = [0, 0, 0, 0]

    pass_explanation = html.Div([
        html.H5("Forward pass (data flows →)"),
        html.Ol([
            html.Li([
                html.B("Linear transform: "),
                f"z¹ = W¹ · x + b¹  (W¹ is {width}×2, b¹ is {width}×1)"
            ]),
            html.Li([
                html.B("Activation: "),
                f"a¹ = {activation}(z¹)  — nonlinearity bends the space"
            ]),
            html.Li([
                html.B("Output: "),
                f"z² = W² · a¹ + b²  →  ŷ = σ(z²)  (probability)"
            ]),
            html.Li([
                html.B("Loss: "),
                "L = −[y·log(ŷ) + (1−y)·log(1−ŷ)]  (binary cross-entropy)"
            ]),
        ], style={'fontSize': '12px'}),

        html.H5("Backward pass (gradients flow ←)"),
        html.Ol([
            html.Li([
                html.B("∂L/∂z² "),
                "= ŷ − y  (output error)"
            ]),
            html.Li([
                html.B("∂L/∂W² "),
                f"= ∂L/∂z² · a¹ᵀ  →  ||∇W²|| = {grad_norms[2]:.4f}"
            ]),
            html.Li([
                html.B("Chain rule into hidden: "),
                f"∂L/∂a¹ = W²ᵀ · ∂L/∂z²  →  "
                f"∂L/∂z¹ = ∂L/∂a¹ ⊙ {activation}'(z¹)"
            ]),
            html.Li([
                html.B("∂L/∂W¹ "),
                f"= ∂L/∂z¹ · xᵀ  →  ||∇W¹|| = {grad_norms[0]:.4f}"
            ]),
        ], style={'fontSize': '12px'}),

        html.P(f"Weight update: W ← W − {lr:.4f} · ∂L/∂W  (gradient descent)",
               style={'fontWeight': 'bold', 'fontSize': '12px', 'marginTop': '8px'}),
    ])

    # 6) Over/under-fitting status
    if len(params['train_loss']) >= 2:
        t_loss = params['train_loss'][-1]
        v_loss = params['val_loss'][-1]
        t_acc = params['train_acc'][-1]
        v_acc = params['val_acc'][-1]
        gap = v_loss - t_loss

        if t_loss > 0.5:
            status_color = '#f97316'
            status_text = 'Under-fitting – the model has not learned the pattern yet. Train more or increase capacity.'
        elif gap > 0.15:
            status_color = '#ef4444'
            status_text = (f'Over-fitting detected – val loss ({v_loss:.3f}) much higher '
                           f'than train loss ({t_loss:.3f}). '
                           'The model memorises training data. Try fewer neurons or early stopping.')
        else:
            status_color = '#22c55e'
            status_text = 'Good fit – training and validation loss are close. The model generalises well.'

        fit_status = html.Div([
            html.Div(status_text,
                     style={'color': status_color, 'fontWeight': 'bold',
                            'marginBottom': '10px'}),
            html.Table([
                html.Tr([html.Th(''), html.Th('Train'), html.Th('Val')]),
                html.Tr([html.Td('Loss'), html.Td(f'{t_loss:.4f}'), html.Td(f'{v_loss:.4f}')]),
                html.Tr([html.Td('Accuracy'), html.Td(f'{t_acc:.2%}'), html.Td(f'{v_acc:.2%}')]),
                html.Tr([html.Td('Gap'), html.Td(colspan=2, children=f'{gap:+.4f}')]),
            ], style={'fontSize': '12px', 'borderCollapse': 'collapse',
                      'width': '100%'}),
            html.Hr(),
            html.P('Tip: a large train-val gap signals overfitting. '
                   'A high loss for both signals underfitting. '
                   'The learning rate controls step size—too large overshoots, '
                   'too small learns slowly.',
                   style={'fontSize': '11px', 'fontStyle': 'italic'}),
        ])
    else:
        fit_status = html.P('Press Train to start learning.',
                            style={'color': '#6b7280'})

    epoch_text = f"Epoch: {epoch}"

    return (fig_boundary, fig_loss, fig_acc, fig_grad,
            pass_explanation, fit_status, epoch_text)

#endregion

#====DECLARATION: CODE HERE IS ASSISTED BY Copilot (GPT-5.4) 22/03/26 - 23/04/26====
#region MODEL FACTORY: Level 5 callbacks/methods to show hidden representations and feature spaces - WORK IN PROGRESS

def l5_init_mlp(hidden_width, depth, activation, dataset):
    """Initialise a 2→H→…→H→1 classifier MLP and store dataset."""
    X, y = load_toy_dataset(dataset)
    rng = np.random.default_rng()

    params = {'depth': depth, 'width': hidden_width}

    # First hidden layer  (2 → hidden_width)
    fan_in = 2
    params['W0'] = rng.normal(0, np.sqrt(2.0 / fan_in),
                              size=(hidden_width, fan_in)).tolist()
    params['b0'] = np.zeros(hidden_width).tolist()

    # Additional hidden layers  (hidden_width → hidden_width)
    for d in range(1, depth):
        params[f'W{d}'] = rng.normal(0, np.sqrt(2.0 / hidden_width),
                                     size=(hidden_width, hidden_width)).tolist()
        params[f'b{d}'] = np.zeros(hidden_width).tolist()

    # Output layer  (hidden_width → 1, sigmoid)
    params[f'W{depth}'] = rng.normal(0, np.sqrt(2.0 / hidden_width),
                                     size=(1, hidden_width)).tolist()
    params[f'b{depth}'] = np.zeros(1).tolist()

    params['loss_history'] = []
    params['epoch'] = 0
    params['X'] = X.tolist()
    params['y'] = y.tolist()
    return params


def l5_forward_all_layers(X, params, activation):
    """
    Forward pass returning activations after every layer (list of (N, dim)).
    Index 0 = input (N,2), then one entry per hidden layer, then output probs (N,1).
    """
    depth = params['depth']
    A = X.T  # (2, N)
    layer_outputs = [X.copy()]  # layer 0 = input

    for d in range(depth):
        W = np.array(params[f'W{d}'])
        b = np.array(params[f'b{d}']).reshape(-1, 1)
        Z = W @ A + b
        A = activation_forward(Z, activation)
        layer_outputs.append(A.T.copy())  # (N, hidden_width)

    # Output layer (sigmoid)
    W_out = np.array(params[f'W{depth}'])
    b_out = np.array(params[f'b{depth}']).reshape(-1, 1)
    Z_out = W_out @ A + b_out
    probs = 1.0 / (1.0 + np.exp(-Z_out))  # (1, N)
    layer_outputs.append(probs.T.copy())  # (N, 1)
    return layer_outputs, probs


def l5_train(params, activation, epochs=50, lr=0.05):
    """Train the deep classifier with BCE loss; returns updated params."""
    X = np.array(params['X'])
    y = np.array(params['y'])
    depth = params['depth']
    N = X.shape[0]
    loss_history = list(params.get('loss_history', []))

    for _ in range(epochs):
        # ── forward ──
        As = [X.T]  # list of layer activations, As[0] = (2, N)
        Zs = []

        A = As[0]
        for d in range(depth):
            W = np.array(params[f'W{d}'])
            b = np.array(params[f'b{d}']).reshape(-1, 1)
            Z = W @ A + b
            Zs.append(Z)
            A = activation_forward(Z, activation)
            As.append(A)

        W_out = np.array(params[f'W{depth}'])
        b_out = np.array(params[f'b{depth}']).reshape(-1, 1)
        Z_out = W_out @ A + b_out
        probs = 1.0 / (1.0 + np.exp(-Z_out))  # (1, N)

        # BCE loss
        eps = 1e-8
        y_row = y.reshape(1, -1).astype(float)
        loss = -float(np.mean(y_row * np.log(probs + eps)
                              + (1 - y_row) * np.log(1 - probs + eps)))
        loss_history.append(loss)

        # ── backward ──
        dZ_out = probs - y_row  # (1, N)

        dW_out = (dZ_out @ As[depth].T) / N
        db_out = np.mean(dZ_out, axis=1, keepdims=True)
        dA = W_out.T @ dZ_out  # (width, N)

        W_out -= lr * dW_out
        b_out -= lr * db_out
        params[f'W{depth}'] = W_out.tolist()
        params[f'b{depth}'] = b_out.flatten().tolist()

        for d in range(depth - 1, -1, -1):
            dZ = activation_backward(dA, Zs[d], activation)
            W = np.array(params[f'W{d}'])
            dW = (dZ @ As[d].T) / N
            db = np.mean(dZ, axis=1, keepdims=True)
            if d > 0:
                dA = W.T @ dZ
            W -= lr * dW
            b_vec = np.array(params[f'b{d}']).reshape(-1, 1)
            b_vec -= lr * db
            params[f'W{d}'] = W.tolist()
            params[f'b{d}'] = b_vec.flatten().tolist()

    params['loss_history'] = loss_history
    params['epoch'] = params.get('epoch', 0) + epochs
    return params


def l5_project_2d(A, method='pca'):
    """Project (N, D) activations to 2D for visualisation. If D<=2, pad/slice."""
    if A.shape[1] == 1:
        return np.column_stack([A[:, 0], np.zeros(A.shape[0])])
    if A.shape[1] == 2:
        return A[:, :2]
    # Simple PCA via SVD (centred)
    mu = A.mean(axis=0)
    Ac = A - mu
    _, _, Vt = np.linalg.svd(Ac, full_matrices=False)
    return Ac @ Vt[:2].T


def l5_linear_separability_score(A_2d, y):
    """
    Fit a simple linear classifier (perceptron-like) on the 2D projection
    and return accuracy.  Uses closed-form logistic regression shortcut:
    pseudo-inverse solution as a quick proxy.
    """
    N = A_2d.shape[0]
    X_aug = np.column_stack([A_2d, np.ones(N)])  # (N, 3)
    y_col = y.reshape(-1, 1).astype(float)
    # Least-squares solution
    try:
        w = np.linalg.lstsq(X_aug, y_col, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.5
    preds = (X_aug @ w >= 0.5).astype(int).flatten()
    return float(np.mean(preds == y))



@app.callback(
    Output('level5-params-store', 'data'),
    Input('level5-dataset-dropdown', 'value'),
    Input('level5-depth-slider', 'value'),
    Input('level5-width-slider', 'value'),
    Input('level5-activation-dropdown', 'value'),
    Input('level5-reset-btn', 'n_clicks'),
)
def update_level5_params(dataset, depth, width, activation, _reset):
    return l5_init_mlp(width, depth, activation, dataset)

@app.callback(
    Output('level5-params-store', 'data', allow_duplicate=True),
    Output('level5-feature-graph', 'figure'),
    Output('level5-boundary-graph', 'figure'),
    Output('level5-loss-graph', 'figure'),
    Output('level5-layer-slider', 'max'),
    Output('level5-layer-slider', 'marks'),
    Output('level5-layer-slider', 'value'),
    Output('level5-layer-explanation', 'children'),
    Output('level5-separability-info', 'children'),
    Output('level5-epoch-counter', 'children'),
    Input('level5-train50-btn', 'n_clicks'),
    Input('level5-train200-btn', 'n_clicks'),
    Input('level5-layer-slider', 'value'),
    State('level5-params-store', 'data'),
    State('level5-activation-dropdown', 'value'),
    prevent_initial_call=True,
)
def update_level5_views(n50, n200, layer_idx, params, activation):
    if params is None:
        raise dash.exceptions.PreventUpdate

    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else ''

    # ── train if a train button was pressed ──
    if 'train50' in trigger:
        params = l5_train(params, activation, epochs=50, lr=0.05)
    elif 'train200' in trigger:
        params = l5_train(params, activation, epochs=200, lr=0.05)

    depth = params['depth']
    X = np.array(params['X'])
    y = np.array(params['y'])

    # Forward to get all layer activations
    layer_outputs, probs = l5_forward_all_layers(X, params, activation)
    # layer_outputs: [input(N,2), hidden1(N,W), ..., hiddenD(N,W), output(N,1)]

    # Update layer slider range: 0..depth+1
    slider_max = depth + 1
    marks = {0: 'Input'}
    for d in range(1, depth + 1):
        marks[d] = f'Hidden {d}'
    marks[depth + 1] = 'Output'

    # Clamp layer_idx
    if layer_idx is None or layer_idx > slider_max:
        layer_idx = 0

    # ── Feature-space scatter for the selected layer ──
    A = layer_outputs[layer_idx]  # (N, dim)
    A_2d = l5_project_2d(A)

    colors = ['#3b82f6' if yi == 0 else '#ef4444' for yi in y]
    fig_feature = go.Figure()
    fig_feature.add_trace(go.Scatter(
        x=A_2d[:, 0], y=A_2d[:, 1], mode='markers',
        marker=dict(color=colors, size=5, opacity=0.7),
        hovertemplate='dim1: %{x:.3f}<br>dim2: %{y:.3f}<extra></extra>',
    ))
    layer_label = marks.get(layer_idx, f'Layer {layer_idx}')
    dim_actual = A.shape[1]
    proj_note = '' if dim_actual <= 2 else f' (PCA from {dim_actual}D)'
    fig_feature.update_layout(
        title=f'Data after {layer_label}{proj_note}',
        xaxis_title='Dimension 1', yaxis_title='Dimension 2',
        template='plotly_white', margin=dict(t=40, b=30),
    )

    # ── Decision boundary in input space ──
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    step = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    _, grid_probs = l5_forward_all_layers(grid, params, activation)
    zz = grid_probs.flatten().reshape(xx.shape)

    fig_boundary = go.Figure()
    fig_boundary.add_trace(go.Contour(
        x=np.arange(x_min, x_max, step), y=np.arange(y_min, y_max, step),
        z=zz, colorscale='RdBu', opacity=0.5,
        showscale=False, contours=dict(showlines=False),
    ))
    fig_boundary.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1], mode='markers',
        marker=dict(color=y, colorscale='RdBu', size=5, line=dict(width=0.5, color='black')),
        hoverinfo='skip',
    ))
    fig_boundary.update_layout(
        title='Decision boundary (input space)',
        template='plotly_white', margin=dict(t=40, b=30),
        xaxis_title='x₁', yaxis_title='x₂',
    )

    # ── Loss curve ──
    loss_hist = params.get('loss_history', [])
    fig_loss = go.Figure()
    if loss_hist:
        fig_loss.add_trace(go.Scatter(
            y=loss_hist, mode='lines', name='BCE Loss',
            line=dict(color='#f59e0b'),
        ))
    fig_loss.update_layout(
        title='Training loss (BCE)',
        xaxis_title='Epoch', yaxis_title='Loss',
        template='plotly_white', margin=dict(t=40, b=30),
    )

    # ── Layer explanation ──
    epoch = params.get('epoch', 0)
    if layer_idx == 0:
        explanation = html.Div([
            html.P("Layer 0 is the raw input — the original 2D coordinates of each data point."),
            html.P("No transformation has been applied yet. The scatter plot shows the dataset "
                   "exactly as it was generated."),
        ])
    elif layer_idx <= depth:
        explanation = html.Div([
            html.P(f"Hidden layer {layer_idx} applies an affine transformation "
                   f"W{layer_idx}·a + b{layer_idx} followed by the {activation} activation."),
            html.P(f"This maps the {A.shape[1]}-dimensional representation into a new "
                   f"{A.shape[1]}D feature space where the network can separate the classes "
                   "more easily."),
            html.P("Each successive layer bends and folds the space further, progressively "
                   "untangling the data until the classes become linearly separable.",
                   style={'fontStyle': 'italic', 'fontSize': '12px'}),
        ])
    else:
        explanation = html.Div([
            html.P("The output layer applies a final linear map W·a + b followed by sigmoid."),
            html.P("This produces a single probability per sample. Values close to 0 or 1 "
                   "indicate confident classification — the network has (hopefully) made "
                   "the classes separable in the previous hidden layers."),
        ])

    # ── Linear separability score ──
    sep_acc = l5_linear_separability_score(A_2d, y)
    if layer_idx == 0:
        sep_msg = html.Div([
            html.P(f"A straight line through the input space achieves "
                   f"{sep_acc * 100:.1f}% accuracy."),
            html.P("For non-linearly-separable datasets (moons, circles) this will be low — "
                   "that's why we need hidden layers!"),
        ])
    elif layer_idx <= depth:
        sep_color = '#16a34a' if sep_acc > 0.9 else '#eab308' if sep_acc > 0.75 else '#dc2626'
        sep_msg = html.Div([
            html.P([
                "Linear separability of this hidden representation: ",
                html.Span(f"{sep_acc * 100:.1f}%",
                          style={'fontWeight': 'bold', 'color': sep_color, 'fontSize': '18px'}),
            ]),
            html.P("Once this approaches ~95%+ the network has successfully untangled "
                   "the data — a simple straight line can now separate the classes."
                   if sep_acc < 0.95 else
                   "The hidden representation is linearly separable! The network has "
                   "successfully learned to disentangle the classes.",
                   style={'fontStyle': 'italic', 'fontSize': '12px'}),
        ])
    else:
        preds = (np.array(probs).flatten() >= 0.5).astype(int)
        net_acc = float(np.mean(preds == y))
        sep_msg = html.Div([
            html.P(f"Network output accuracy: {net_acc * 100:.1f}%"),
            html.P("This is the final classification. If accuracy is high the preceding "
                   "hidden layers successfully created a linearly-separable feature space."),
        ])

    epoch_text = f"Epoch: {epoch}"

    return (params, fig_feature, fig_boundary, fig_loss,
            slider_max, marks, layer_idx,
            explanation, sep_msg, epoch_text)
#endregion

if __name__ == '__main__':
    app.run(debug=False)   #changed debug to false because otherwise it resets the page every minute


