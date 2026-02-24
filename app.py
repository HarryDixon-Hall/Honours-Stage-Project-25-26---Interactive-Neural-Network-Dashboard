import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
from dataload import load_dataset, get_dataset_stats
from trainer import train_model
from trainer import build_model

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
 
# Load data once at startup
X_train_full, X_test, y_train_full, y_test, meta = load_dataset("iris") #feature/class names removed because meta fulfills those variables

class_names = meta['class_names']
dataset_stats = get_dataset_stats(X_train_full, y_train_full) #can now make use of this with metadata

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)
 
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
            "margin": "0 20px",
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
            dcc.Link("Level 1", href="/level1", 
                    style={'padding': '10px 15px', 'display': 'inline-block', 
                          'color': '#333', 'textDecoration': 'none'}),
            dcc.Link("Level 2", href="/level2", 
                    style={'padding': '10px 15px', 'display': 'inline-block', 
                          'color': '#333', 'textDecoration': 'none'}),
            dcc.Link("Level 3", href="/level3", 
                    style={'padding': '10px 15px', 'display': 'inline-block', 
                          'color': '#333', 'textDecoration': 'none'}),
            dcc.Link("Level 4", href="/level4", 
                    style={'padding': '10px 15px', 'display': 'inline-block', 
                          'color': '#333', 'textDecoration': 'none'}),
            dcc.Link("Level 5", href="/level5", 
                    style={'padding': '10px 15px', 'display': 'inline-block', 
                          'color': '#333', 'textDecoration': 'none'}),
            
        ], style={'float': 'right'})
    ], style={
        'backgroundColor': '#f8f9fa', 
        'borderBottom': '1px solid #dee2e6',
        'padding': '15px 0',
        'position': 'sticky',
        'top': '0',
        'zIndex': '1000'
    }),


    dcc.Location(id="url", refresh=False), #url watchdog
    dcc.Store(id="lesson-config-store"),   #sharing between teacher/student of lesson config
    html.Div(id="page-content")            #student/teacher pa
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


if __name__ == '__main__':
    app.run(debug=False)   #changed debug to false because otherwise it resets the page every minute


