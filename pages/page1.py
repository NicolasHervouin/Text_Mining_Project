from dash import html, dcc, register_page, Input, Output, callback
from utils.def_utils import polarity_evolution, polarity_evolution2, compare_youtubeurs, all_comments, compare_youtubeurs2
import os

API_KEY = os.getenv("API_KEY")
register_page(__name__, path="/page1")

df = all_comments()

layout = html.Div([ 
    html.H2("DashBoard 1: Vision Générale", style={'text-align': 'center'}),
    
    # Menu déroulant pour choisir le youtubeur 
    html.Div([
        dcc.Dropdown(id='youtubeur-dropdown',
                    options= [{'label': youtuber, 'value': youtuber} for youtuber in df['youtubeur'].unique()],
                    value='SQUEEZIE',
                    multi=False,
                    clearable=False,
                    style={"width": "40%"})
    ]),
    
    # Tabs pour afficher les graphiques et les infos de la vidéo
    dcc.Tabs(id='tabs-content', children=[
        dcc.Tab(label='Graph 1: Evolution de la polarité des commentaires', children=[
            dcc.Graph(id='graph1-page2'),
            dcc.Graph(id='graph2-page2')
        ]),
        
        dcc.Tab(label='Graph 2: Comparaison des youtubeurs', children=[
            dcc.Graph(id='graph3-page2'),
            dcc.Graph(id='graph4-page2')
        ]),
        
        dcc.Tab(label='Graph 3: Association de mots', children=[
            html.Img(id='image1-page2'),
            html.Img(id='image2-page2')
            
        ]), 
        
        dcc.Tab(label='Graph 4: Nuages de mots', children=[
            html.Img(id='image3-page2'),
            html.Img(id='image4-page2')
            
        ]),        
    ])
])  
@callback(
    Output('graph1-page2', 'figure'),
    Output('graph2-page2', 'figure'),     
    Output('graph3-page2', 'figure'), 
    Output('graph4-page2', 'figure'),    
    Output('image1-page2', 'src'), 
    Output('image2-page2', 'src'),
    Output('image3-page2', 'src'),    
    Output('image4-page2', 'src'),
    Input('youtubeur-dropdown', 'value')
)
def update_graphs(youtubeur):
    df_youtubeur = df[df['youtubeur'] == youtubeur]
    
    # Graph 1: Evolution de la polarité des vidéos
    fig1 = polarity_evolution(df_youtubeur)
    fig2 = polarity_evolution2(df_youtubeur)
    
    # Graph 2: Comparaison des youtubeurs
    fig3 = compare_youtubeurs(df,youtubeur)
    fig4 = compare_youtubeurs2(df,youtubeur)
    
    # Graph 3: Association de mots
    fig5 = '/assets/images/bigramme.png'
    fig6 = '/assets/images/trigramme.png'
    
    # Graph 4: Nuages de mots
    fig7 = '/assets/images/wordcloud_vpnpos.png'
    fig8 = '/assets/images/wordcloud_vpnneg.png'
    
    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8


