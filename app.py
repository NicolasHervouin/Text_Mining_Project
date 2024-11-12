from dash import Dash, dcc, html, page_container, callback, Input, Output

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    html.Div([
        html.Div(id='nav-bar', style={
            'width': '350px', 
            'padding': '20px', 
            'background-color': '#f8f9fa', 
            'position': 'fixed', 
            'top': '0', 
            'left': '0', 
            'bottom': '0',
            'overflow': 'auto',
        }),
        
        html.Div([
            page_container
        ], style={'margin-left': '370px', 'padding': '20px'}) 
    ])
])

@app.callback(
    Output('nav-bar', 'children'),
    [Input('url', 'pathname')] 
)
def update_navbar(pathname):
    return html.Div([
        # Ajout du logo au-dessus du titre
        html.Img(src='/assets/Logo-NordVPN.png', style={'width': '100%', 'margin-bottom': '20px'}),  

        dcc.Link(
            html.H1("Analyse Des Sponsoring YouTube", style={'text-align': 'center', 'font-family': 'Arial, sans-serif'}),
            href='/',
            style={'text-decoration': 'none'}    
        ),
        dcc.Link('A venir', href='/page1', className='nav-link nav-link-active' if pathname == '/page1' else 'nav-link', style={'display': 'block', 'margin-bottom': '10px'}),
        dcc.Link('Search a video', href='/page2', className='nav-link nav-link-active' if pathname == '/page2' else 'nav-link', style={'display': 'block', 'margin-bottom': '10px'}),
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
