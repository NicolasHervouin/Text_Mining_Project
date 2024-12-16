from dash import html, dcc, register_page, Input, Output, callback
from utils.def_api_google import obtenir_infos_video_et_commentaires
from utils.def_utils import polarity_evolution,polarity_evolution2, compare_youtubeurs, word_cloud, polarity_plot, subjectivity_plot, polarity_on_vpn, all_comments
import os
import pandas as pd
import time
import spacy
import matplotlib.pyplot as plt

API_KEY = os.getenv("API_KEY")
register_page(__name__, path="/page1")
nlp = spacy.load("fr_core_news_sm")
df = all_comments()
print(df.head())

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
            dcc.Graph(id='graph3-page2')
        ]),
        
        dcc.Tab(label='Graph 3: Nuages de mots', children=[
            # html.Div(id='graph4-page2')
            # html.Div(id='graph5-page2')
            html.Div(id='image1-page2'),
            html.Div(id='image2-page2')
            
        ]),
        
    ])
])  
@callback(
    Output('graph1-page2', 'figure'),
    Output('graph2-page2', 'figure'),     
    Output('graph3-page2', 'figure'), 
    # Output('graph4-page2', 'figure'),    
    # Output('graph5-page2', 'figure'), 
    Output('image1-page2', 'src'),    
    Output('image2-page2', 'src'),
    Input('youtubeur-dropdown', 'value')
)
def update_graphs(youtubeur):
    df_youtubeur = df[df['youtubeur'] == youtubeur]
    
    # Graph 1: Evolution de la polarité des vidéos
    fig1 = polarity_evolution(df_youtubeur)
    fig2 = polarity_evolution2(df_youtubeur)
    
    # Graph 2: Comparaison des youtubeurs
    fig3 = compare_youtubeurs(df,youtubeur)
    
    # Graph 3: Nuages de mots
    df_polarity_pos = df[df['Number'] > 0]
    df_polarity_neg = df[df['Number'] < 0]    
    # fig4 = word_cloud(df_polarity_pos, nlp, keyword_sheet_id='pola_pos', want_to_save=True)
    # fig5 = word_cloud(df_polarity_neg, nlp, keyword_sheet_id='pola_neg', want_to_save=True)
    
    fig4 = html.Img(src='data/images/wordcloud_vpnpos.png', style={'width': '50%', 'display': 'inline-block'})
    fig5 = html.Img(src='data/images/wordcloud_vpnneg.png', style={'width': '50%', 'display': 'inline-block'})
    
    
    return fig1, fig2, fig3, fig4, fig5


