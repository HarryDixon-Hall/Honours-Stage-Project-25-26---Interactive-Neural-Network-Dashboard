from dash import dcc, html


#region ADAPTIVE LEARNING: hopefully this can be be integrated with gamification features
#skill tree data as a placeholder to showcase clicing on different levels at different points
SKILL_TREE_DATA = {
    "nodes": [
        {"id": "level1", "name": "Hyperparams", "x": 0, "y": 0, "unlocked": True, "completed": False},
        {"id": "level2", "name": "Templates", "x": 1, "y": 0, "unlocked": False, "completed": False},
        {"id": "level3", "name": "Functions", "x": 2, "y": 0, "unlocked": False, "completed": False},
        {"id": "level4", "name": "Classes", "x": 1, "y": 1, "unlocked": False, "completed": False},
        {"id": "level5", "name": "Optimisers", "x": 2, "y": 1, "unlocked": False, "completed": False},
    ],
    "prereqs": {
        "level2": ["level1"],
        "level3": ["level2"], 
        "level4": ["level2"],
        "level5": ["level3", "level4"]
    }
}
def skilltree_layout():
    return html.Div([
        html.H1("SKILL TREE",
                    style={'textAlign': 'center', 'marginBottom': '20px'}),

        html.Div([
            dcc.Link([skill_box("Level 3", "Build a FNN model by Code")], href="/level3"),       # row 1

            dcc.Link([skill_box("Level 2", "Build a FNN model by UI")], href="/level2"),       # row 2

            dcc.Link([skill_box("Level 1", "Preconfigured FFNN Explorer")], href="/level1"), # row 3
        ])            
    ], style={
            "display": "grid",
            "gridTemplateColumns": "repeat(1, 1fr)",
            "gap": "20px",
            "maxWidth": "900px",
            "margin": "0 auto",
            "padding": "20px"
    })
def skill_box(title, subtitle):
    return html.Div([
        html.H3(title),
        html.P(subtitle)
    ], className = "skill-box")
#endregion