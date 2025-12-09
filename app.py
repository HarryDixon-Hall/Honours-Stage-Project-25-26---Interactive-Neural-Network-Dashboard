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
 
@app.callback(
    [Output('training-curves', 'figure'),
     Output('test-accuracy', 'children'),
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
   
    # Test accuracy
    test_output = model.forward(X_test)
    test_preds = np.argmax(test_output, axis=1)
    test_acc = np.mean(test_preds == y_test)
   
    # Create training curves figure
    fig = go.Figure()
   
    fig.add_trace(go.Scatter(
        y=history['loss'],
        mode='lines',
        name='Loss',
        line=dict(color='#FF6B6B', width=2)
    ))
   
    fig.add_trace(go.Scatter(
        y=history['accuracy'],
        mode='lines',
        name='Accuracy',
        line=dict(color='#4ECDC4', width=2),
        yaxis='y2'
    ))
   
    fig.update_layout(
        title='Training Progress',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        yaxis2=dict(title='Accuracy', overlaying='y', side='right'),
        hovermode='x unified',
        height=400
    )
   
    status_msg = f"✓ Training complete! (Config: {int(hidden_size)}-neuron hidden layer, LR={learning_rate:.4f})"
    accuracy_msg = f"Test Accuracy: {test_acc:.2%}"
   
    return fig, accuracy_msg, status_msg, {'loss': history['loss'], 'accuracy': history['accuracy']}
 
if __name__ == '__main__':
    app.run(debug=False)   #changed debug to false because otherwise it resets the page every minute