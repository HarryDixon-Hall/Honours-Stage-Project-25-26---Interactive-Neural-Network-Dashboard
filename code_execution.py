import base64
import io
import sys

import plotly.graph_objects as go
from dash import html


def execute_python_snippet(user_code, safe_env, plt_module=None):
    output_capture = io.StringIO()

    try:
        old_stdout = sys.stdout
        sys.stdout = output_capture
        exec(user_code, safe_env)
        sys.stdout = old_stdout

        outputs = []
        if plt_module is not None and plt_module.get_fignums():
            figure = plt_module.gcf()
            buffer = io.BytesIO()
            figure.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_string = base64.b64encode(buffer.read()).decode()
            plt_module.close('all')
            outputs.append(
                html.Img(
                    src=f'data:image/png;base64,{image_string}',
                    style={
                        'max-width': '100%',
                        'height': 'auto',
                        'display': 'block',
                        'margin-bottom': '10px',
                    },
                )
            )

        result = output_capture.getvalue()
        if result.strip():
            outputs.append(
                html.Pre(
                    result,
                    style={
                        'margin': '0',
                        'white-space': 'pre-wrap',
                        'font-family': 'monospace',
                        'font-size': '14px',
                    },
                )
            )

        if not outputs:
            outputs.append(html.Pre('Code executed successfully (no output)'))

        return html.Div(outputs), '', go.Figure()
    except SyntaxError as exc:
        sys.stdout = old_stdout
        return (
            '',
            html.Div([
                html.H3(f'Syntax Error (Line {exc.lineno})', style={'color': 'red'}),
                html.Pre(str(exc)),
            ]),
            go.Figure(),
        )
    except Exception as exc:
        sys.stdout = old_stdout
        return (
            '',
            html.Div([
                html.H3('Compile Error', style={'color': 'red'}),
                html.Pre(str(exc)),
            ]),
            go.Figure(),
        )