import dash
from dash import dcc, html
import webbrowser
import threading

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Proof of Concept"),
    html.P("This executable uses Dash and Plotly"),
    dcc.Graph(
        figure={
            'data': [
                {'x': [1, 2, 3, 4], 'y': [4, 1, 3, 2], 'type': 'bar', 'name': 'Test Data'}
            ],
            'layout': {'title': 'Simple Chart'}
        }
    )
])

if __name__ == '__main__':


    threading.Timer(1.0, lambda: webbrowser.open('http://localhost:8050')).start()
    app.run(debug=True)