from dash import html


def home_layout(): #removed the type error by splitting it out
    return html.Div([
            html.H2("HOME PAGE: WELCOME TO NEURAL NETWORKS",
                    style={'textAlign': 'center', 'marginBottom': '20px'}),
            
            html.H2("Welcome to Neural Network Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            html.H2("Skill tree to access levels, sandbox to see python coding enviroment.", 
                   style={'textAlign': 'center', 'fontSize': '18px'}),
                   #INTRODUCTION_TEXT  # Reuse existing intro content (maybe)
            
    ])
#endregion