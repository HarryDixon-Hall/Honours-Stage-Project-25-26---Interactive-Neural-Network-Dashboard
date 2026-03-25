from dash import html

from code_execution import CodeExecutionBox


SANDBOX_EDITOR = CodeExecutionBox(
    'sandbox-editor',
    ids={
        'input': 'code-input',
        'run': 'code-run',
        'export': 'code-export',
        'download': 'download-editor',
        'output': 'editor-output',
        'error': 'editor-error',
        'plot': 'editor-plot',
        'validation': 'code-validation',
        'highlighted': 'code-highlighted',
    },
)


#region SECURE PROGRAMMING: layout for sandbox code
def sandbox_layout():
    return html.Div([
        html.H2("SANDBOX",
                style={'textAlign': 'center', 'marginBottom': '20px'}),
        #instructions for ML pipeline
        html.Div([
            html.H3("Build your own ML pipeline from scratch."),
            html.H2("1. Load data"),
            html.H2("2. Feature engineering"),
            html.H2("3. Train Model"),
            html.H2("4. Evaluate"),
        ]),
        
        html.Div([
            html.Label("Python Code Editor:", style={"fontWeight": "bold"}),
            SANDBOX_EDITOR.render(
                default_code="print('Hello World')",
                title=None,
                run_label='Run Code',
                export_label='Export Code',
                show_export=True,
                include_plot=True,
                code_height='400px',
                wrapper_style={'padding': '0', 'boxShadow': 'none', 'backgroundColor': 'transparent'},
            )
        ], style={"maxWidth": "1400px", "margin": "0 auto"})


    ])
#endregion
