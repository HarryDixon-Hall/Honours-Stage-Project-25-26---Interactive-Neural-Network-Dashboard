import ast
import base64
import io
import keyword
import sys
import tokenize
from io import StringIO

from dash import dcc, html
import plotly.graph_objects as go


class CodeExecutionBox:
    def __init__(self, prefix, ids=None):
        default_ids = {
            'input': f'{prefix}-input',
            'run': f'{prefix}-run',
            'export': f'{prefix}-export',
            'download': f'{prefix}-download',
            'output': f'{prefix}-output',
            'error': f'{prefix}-error',
            'plot': f'{prefix}-plot',
            'validation': f'{prefix}-validation',
            'highlighted': f'{prefix}-highlighted',
        }
        if ids:
            default_ids.update(ids)
        self.ids = default_ids

    def render(
        self,
        *,
        default_code,
        title=None,
        description=None,
        controls=None,
        run_label='Run Code',
        export_label='Export Code',
        show_export=False,
        include_plot=True,
        code_height='320px',
        output_placeholder='',
        wrapper_style=None,
    ):
        if controls is None:
            controls = []

        children = []
        if title or run_label:
            header_children = []
            if title:
                header_children.append(html.Span(title, style={'fontWeight': '700', 'fontSize': '16px'}))
            header_children.append(
                html.Button(
                    run_label,
                    id=self.ids['run'],
                    n_clicks=0,
                    style={
                        'backgroundColor': '#0f766e',
                        'color': 'white',
                        'border': 'none',
                        'padding': '8px 12px',
                        'borderRadius': '8px',
                    },
                )
            )
            children.append(
                html.Div(
                    header_children,
                    style={
                        'display': 'flex',
                        'justifyContent': 'space-between',
                        'alignItems': 'center',
                        'marginBottom': '10px',
                    },
                )
            )

        if description:
            children.append(html.P(description, style={'fontSize': '13px', 'color': '#475569', 'marginBottom': '12px'}))

        children.append(
            dcc.Textarea(
                id=self.ids['input'],
                value=default_code,
                style={
                    'width': '100%',
                    'height': code_height,
                    'fontFamily': 'Consolas, Monaco, monospace',
                    'fontSize': '13px',
                    'lineHeight': '1.5',
                    'backgroundColor': '#0f172a',
                    'color': '#e2e8f0',
                    'border': 'none',
                    'borderRadius': '10px',
                    'padding': '12px',
                    'boxSizing': 'border-box',
                    'marginBottom': '12px',
                },
            )
        )

        children.append(
            html.Div(
                id=self.ids['validation'],
                style={
                    'marginBottom': '12px',
                    'fontSize': '12px',
                    'fontWeight': '600',
                },
            )
        )

        if controls:
            children.append(html.Div(controls, style={'marginBottom': '12px'}))

        if show_export:
            children.append(
                html.Div(
                    [
                        html.Button(
                            export_label,
                            id=self.ids['export'],
                            n_clicks=0,
                            style={
                                'width': '100%',
                                'padding': '12px',
                                'fontSize': '16px',
                            },
                        ),
                        dcc.Download(id=self.ids['download']),
                    ],
                    style={'marginBottom': '12px'},
                )
            )

        children.append(
            html.Details(
                [
                    html.Summary('Syntax Highlighted Preview', style={'cursor': 'pointer', 'fontWeight': '600'}),
                    html.Div(
                        id=self.ids['highlighted'],
                        style={
                            'marginTop': '10px',
                            'backgroundColor': '#f8fafc',
                            'border': '1px solid #dbeafe',
                            'borderRadius': '10px',
                            'padding': '12px',
                        },
                    ),
                ],
                style={'marginBottom': '12px'},
            )
        )

        children.append(html.Div(output_placeholder, id=self.ids['output']))
        children.append(html.Div(id=self.ids['error'], style={'color': '#dc3545'}))
        if include_plot:
            children.append(dcc.Graph(id=self.ids['plot']))

        default_wrapper_style = {
            'backgroundColor': 'white',
            'borderRadius': '16px',
            'padding': '18px',
            'boxShadow': '0 2px 10px rgba(15, 23, 42, 0.08)',
        }
        if wrapper_style:
            default_wrapper_style.update(wrapper_style)
        return html.Div(children, style=default_wrapper_style)

    def build_validation_message(self, code):
        if not code or not code.strip():
            return html.Span('No code to validate.', style={'color': '#64748b'})

        try:
            ast.parse(code)
            return html.Span('Syntax valid', style={'color': '#15803d'})
        except SyntaxError as exc:
            return html.Span(f'Syntax error on line {exc.lineno}: {exc.msg}', style={'color': '#b91c1c'})

    def build_highlighted_code(self, code):
        if not code:
            return html.Pre('', style={'margin': '0', 'whiteSpace': 'pre-wrap'})

        children = []
        line_number = 1
        column_number = 0

        try:
            tokens = tokenize.generate_tokens(StringIO(code).readline)
            for token_type, token_string, start, end, _ in tokens:
                if token_type == tokenize.ENDMARKER:
                    break

                start_row, start_col = start
                end_row, end_col = end

                if start_row > line_number:
                    children.append('\n' * (start_row - line_number))
                    line_number = start_row
                    column_number = 0

                if start_col > column_number:
                    children.append(' ' * (start_col - column_number))

                token_style = self._style_for_token(token_type, token_string)
                if token_style:
                    children.append(html.Span(token_string, style=token_style))
                else:
                    children.append(token_string)

                line_number = end_row
                column_number = end_col
        except tokenize.TokenError:
            children = [code]

        return html.Pre(
            children,
            style={
                'margin': '0',
                'whiteSpace': 'pre-wrap',
                'fontFamily': 'Consolas, Monaco, monospace',
            },
        )

    def _style_for_token(self, token_type, token_string):
        if token_type == tokenize.COMMENT:
            return {'color': '#64748b', 'fontStyle': 'italic'}
        if token_type == tokenize.STRING:
            return {'color': '#16a34a'}
        if token_type == tokenize.NUMBER:
            return {'color': '#d97706'}
        if token_type == tokenize.OP:
            return {'color': '#db2777'}
        if token_type == tokenize.NAME and token_string in keyword.kwlist:
            return {'color': '#2563eb', 'fontWeight': '700'}
        return None


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