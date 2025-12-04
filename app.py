import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
from dataload import load_dataset_iris, get_dataset_stats
from trainer import train_model
 
# Load data once at startup
X_train, X_test, y_train, y_test, feature_names, class_names = load_dataset_iris()
dataset_stats = get_dataset_stats(X_train, y_train)
 
app = dash.Dash(__name__)
 
# App layout
app.layout = html.Div([
    html.H1("Neural Network Visualizer", style={'textAlign': 'center', 'marginBottom': 30}),
   
    html.Div([
        # LEFT: Controls
        html.Div([
            html.H3("Configuration"),
           
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
                html.P(f"Samples: {dataset_stats['samples']} | Features: {dataset_stats['features']} | Classes: {dataset_stats['classes']}")
            ], style={'marginBottom': 20, 'padding': '10px', 'backgroundColor': '#e8f4f8', 'borderRadius': '5px'}),
           
            # Training curves
            dcc.Graph(id='training-curves'),
           
            # Test accuracy
            html.Div(id='test-accuracy', style={'marginTop': 10, 'fontSize': 16, 'fontWeight': 'bold'})
           
        ], style={'width': '76%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'}),
    ]),
   
    # Hidden store for model history
    dcc.Store(id='model-history-store')
])

 
if __name__ == '__main__':
    app.run_server(debug=True)