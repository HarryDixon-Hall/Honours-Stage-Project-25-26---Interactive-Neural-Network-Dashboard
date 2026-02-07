import dash
from dash import dcc, html, Input, Output, State
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
#2.2 Bottom left - Hyperparameter Control panel (FNN architecture config and Training setup config)
#2.3 Top right - Feed forward neural network architecture
#2.4 Bottom right - Training outcomes (loss graphs, confusion matrices, accuracy metrics etc)

#3. Callbacks
#3.1 Information box
#3.2 Hyperparameter Control panel
#3.3 Reset trained model


#Information that could be used in the callback of the information box
INTRODUCTION_TEXT = """
This dashboard provides an interactive walkthrough of a Feed-Forward Neural Network solving a classification problem with Iris dataset.
"""

THEORY_TEXT = """
Placeholder text
"""

TASKS_TEXT = """
Placeholder text
"""

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
                #2.2 Bottom left - Hyperparameter Control panel (FNN architecture config and Training setup config)
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
                                    "Model Hyperparameters",
                                    style={"fontSize": "14px", "marginBottom": "8px"},
                                ),
                                html.Label("Weight Initialization Seed:", style={"fontSize": "12px"}),
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
                                html.Label("Hidden Layer Size:", style={"fontSize": "12px"}),
                                dcc.Slider(
                                    id="hidden-size",
                                    min=4,
                                    max=64,
                                    step=4,
                                    value=8,
                                    marks={4: "4", 16: "16", 32: "32", 64: "64"},
                                ),
                                html.Div(style={"height": "10px"}),
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
                                html.Div(style={"height": "10px"}),
                                html.Label("Epochs:", style={"fontSize": "12px"}),
                                dcc.Slider(
                                    id="epochs",
                                    min=10,
                                    max=100,
                                    step=10,
                                    value=50,
                                    marks={10: "10", 50: "50", 100: "100"},
                                ),
                                html.Div(style={"height": "16px"}),
                                html.H4(
                                    "Training Setup Notes",
                                    style={"fontSize": "14px", "marginBottom": "4px"},
                                ),
                                html.P(
                                    "Iterative re-training with different hyperparameters. "
                                    "Use Reset to clear stored history if you choose to start over.",
                                    style={"fontSize": "12px", "color": "#4b5563"},
                                ),
                            ],
                        ),
                    ],
                ),


                #=========
                #2.3 Top right - Feed forward neural network architecture
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
                #2.4 Bottom right - Training outcomes (loss graphs, confusion matrices, accuracy metrics etc)
                #=========

            ]

        
        )
    ]
)

"""
app.layout = html.Div([
    html.H1("Neural Network Visualizer", style={'textAlign': 'center', 'marginBottom': 30}),
   
    html.Div([
        # LEFT: Controls
        html.Div([
            html.H3("Configuration"),

            html.Label("Weight initalisation seed:"),
            dcc.Input(id='seed-input', type='number', value=42,
                      style={'width': '100%', 'padding': '8px', 'marginBottom': '20px'}),
           
            html.Label("Hidden Layer Size:"),
            dcc.Slider(id='hidden-size', min=4, max=64, step=4, value=8,
                      marks={4: '4', 16: '16', 32: '32', 64: '64'}),
           
            html.Label("Learning Rate:", style={'marginTop': 20}),
            dcc.Slider(id='learning-rate', min=-3, max=-1, step=0.1, value=-2,
                      marks={-3: '0.001', -2: '0.01', -1: '0.1'},
                      tooltip={'placement': 'bottom', 'always_visible': True}),
           
            html.Label("Epochs:", style={'marginTop': 20}),
            dcc.Slider(id='epochs', min=10, max=100, step=10, value=50,
                      marks={10: '10', 50: '50', 100: '100'}),
           
            html.Button('Train Model', id='train-btn', n_clicks=0,
                       style={'marginTop': 30, 'padding': '10px 20px', 'width': '100%'}),
           
            html.Div(id='status-text', style={'marginTop': 10, 'color': 'green'}),
           
        ], style={'width': '22%', 'display': 'inline-block', 'verticalAlign': 'top',
                 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px'}),
       
        # RIGHT: Visualizations
        html.Div([
            # Dataset info
            html.Div([
                html.H4("Dataset Info"),
                html.P(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test {X_test.shape[0]} | Features {X_train.shape[1]} | Classes: 3")
            ], style={'marginBottom': 20, 'padding': '10px', 'backgroundColor': '#e8f4f8', 'borderRadius': '5px'}),
           
            # Training curves
            dcc.Graph(id='training-curves', style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'}),

            #architecture graph
            dcc.Graph(id='architecture-graph', style={'marginTop': 20, 'height': 350}),

            #heat map graph
            dcc.Graph(id='confusion-matrix-heatmap', style={'marginTop': 20, 'height': 300}),

            #per-class metrics
            html.Div(id='per-class-metrics', style={'marginTop': 20, 'height': 400}),

            #accuracy metrics box
            html.Div(id='accuracy-metrics', style={'marginTop': 20, 'height': 400, 'width': '76%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'}),
        ]),
           
    ]),
   
    # Hidden store for model history
    dcc.Store(id='model-history-store')

])
"""
 
@app.callback(
    [Output('training-curves', 'figure'),
     Output('architecture-graph', 'figure'),
     Output('confusion-matrix-heatmap', 'figure'),
     Output('accuracy-metrics', 'children'),
     Output('per-class-metrics', 'children'),
     Output('status-text', 'children'),
     Output('model-history-store', 'data')],
    Input('train-btn', 'n_clicks'),
    [State('seed-input', 'value'),
     State('hidden-size', 'value'),
     State('learning-rate', 'value'),
     State('epochs', 'value')],
    prevent_initial_call=True
)
def train_and_visualize(n_clicks, seed, hidden_size, learning_rate_log, epochs):
    """Train model and update visualizations"""

    #Seed weight validation
    if seed is None:
        seed = 42
    seed = int(seed)
   
    # Convert log scale back to linear
    learning_rate = 10 ** learning_rate_log
   
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
   
    status_msg = f"✓ Training complete! (Seed={seed}, {int(hidden_size)}-neuron, LR={learning_rate:.4f})"
    #accuracy_msg = f"Test Accuracy: {test_acc:.2%}"
    return train_fig, arch_fig, cm_fig, per_class_metrics, accuracy_metrics, status_msg, {
        'train_loss': history['loss'],
        'train_acc': history['accuracy'],
        'val_loss': val_history['loss'],
        'val_acc': val_history['accuracy']
    }
   

 
if __name__ == '__main__':
    app.run(debug=False)   #changed debug to false because otherwise it resets the page every minute


