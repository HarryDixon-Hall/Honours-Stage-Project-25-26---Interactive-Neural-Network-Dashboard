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


# Load data once at startup
X_train_full, X_test, y_train_full, y_test, meta = load_dataset("iris") #feature/class names removed because meta fulfills those variables

class_names = meta['class_names']
dataset_stats = get_dataset_stats(X_train_full, y_train_full) #can now make use of this with metadata

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

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
 
#callback for training outcomes
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
   


 
 #start app

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

#callback for visualisation of UI sliders in real-time - to see the FNN architecture for the page layouts

# REPLACE your callback with this (app.py)
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

# NEW: Self-contained decision boundary (app.py)
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

#level 2 callback

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

def make_network_diagram_figure(input_dim=2, hidden_dim=4, output_dim=1):
    # x positions for 3 layers
    x_in, x_hid, x_out = 0, 1, 2

    # y positions: spread nodes vertically
    y_in = np.linspace(0, 1, input_dim)
    y_hid = np.linspace(0, 1, hidden_dim)
    y_out = np.linspace(0, 1, output_dim)

    nodes_x = []
    nodes_y = []
    text = []
    layer_colors = []

    # Input nodes
    for i in range(input_dim):
        nodes_x.append(x_in)
        nodes_y.append(y_in[i])
        text.append(f"x{i+1}")
        layer_colors.append("lightblue")

    # Hidden nodes
    for j in range(hidden_dim):
        nodes_x.append(x_hid)
        nodes_y.append(y_hid[j])
        text.append(f"h{j+1}")
        layer_colors.append("lightgreen")

    # Output nodes
    for k in range(output_dim):
        nodes_x.append(x_out)
        nodes_y.append(y_out[k])
        text.append(f"ŷ")
        layer_colors.append("salmon")

    node_trace = go.Scatter(
        x=nodes_x,
        y=nodes_y,
        mode='markers+text',
        text=text,
        textposition='middle right',
        marker=dict(size=18, color=layer_colors, line=dict(width=1, color='black'))
    )

    # Edges as separate Scatter traces (lines)
    edge_traces = []

    # Input -> Hidden
    for i in range(input_dim):
        for j in range(hidden_dim):
            edge_traces.append(go.Scatter(
                x=[x_in, x_hid],
                y=[y_in[i], y_hid[j]],
                mode='lines',
                line=dict(color='grey', width=1),
                hoverinfo='skip',
                showlegend=False
            ))

    # Hidden -> Output
    for j in range(hidden_dim):
        for k in range(output_dim):
            edge_traces.append(go.Scatter(
                x=[x_hid, x_out],
                y=[y_hid[j], y_out[k]],
                mode='lines',
                line=dict(color='grey', width=1),
                hoverinfo='skip',
                showlegend=False
            ))

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Single hidden layer network",
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

    # 2) Activation function figure
    fig_activation = make_activation_figure(activation)

    # 3) Network diagram – e.g. nodes arranged in 3 columns (input, hidden, output)
    fig_network = make_network_diagram_figure(input_dim=2, hidden_dim=width, output_dim=1)

    # 4) Math explanation – show Wx + b and ρ(Wx + b), with parameter count
    n_in, n_h, n_out = 2, width, 1
    num_params = (n_in + 1) * n_h + (n_h + 1) * n_out
    explanation = html.Div([
        html.P(f"Layer 1 (hidden): z¹ = W¹ x + b¹, a¹ = ρ(z¹) with W¹ ∈ ℝ^{n_h}×{n_in}, b¹ ∈ ℝ^{n_h}."),
        html.P(f"Layer 2 (output): z² = W² a¹ + b² with W² ∈ ℝ^{n_out}×{n_h}, b² ∈ ℝ^{n_out}."),
        html.P(f"Total trainable parameters: {num_params}."),
        html.P(f"Current activation: ρ(z) = {activation}."),
    ])

    return fig_boundary, fig_activation, fig_network, explanation



if __name__ == '__main__':
    app.run(debug=False)   #changed debug to false because otherwise it resets the page every minute


