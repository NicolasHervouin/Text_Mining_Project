import os
import pandas as pd
import spacy
from dash import html, dcc, register_page, Input, Output, callback
from utils.def_api_google import obtenir_infos_video_et_commentaires
from utils.def_utils import get_lang_detector, comment_analysis, word_cloud, polarity_plot, subjectivity_plot, polarity_on_vpn
import time
import json

API_KEY = os.getenv("API_KEY")
register_page(__name__, path="/page2")

# Préparation du modèle
# nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("fr_core_news_sm")
nlp.add_pipe("spacytextblob", last=True)
get_lang_detector(nlp, "language_detector")
nlp.add_pipe("language_detector")

# Layout de la page
layout = html.Div([
    html.H2("DashBoard 2: Youtube Comments Analysis"),
    
    # Input pour entrer le VIDEO_ID et bouton pour lancer l'analyse
    html.Div([
        dcc.Input(id='video-id-input', type='text', placeholder='Enter Video ID'),
        html.Button('Retrieve Comments', id='retrieve-comments-button', n_clicks=0),
        html.Div(id='status-message', style={'marginTop': '10px', 'color': 'blue'})
    ], style={'marginBottom': '20px'}),
    
    # Tabs pour afficher les graphiques et les infos de la vidéo
    dcc.Tabs(id='tabs-content', children=[
        dcc.Tab(label='Video Information', children=[
            html.Div(id='video-info-content')
        ]),
        
        dcc.Tab(label='Graph 1: WordCloud', children=[
            dcc.Graph(id='graph1-page1')
        ]),
        
        dcc.Tab(label='Graph 2: Polarity plot', children=[
            dcc.Graph(id='graph2-page1')
        ]),
        
        dcc.Tab(label='Graph 3: Subjectivity plot', children=[
            dcc.Graph(id='graph3-page1')
        ]),
        
        dcc.Tab(label='Graph 4: Gauge de polarité VPN', children=[
            dcc.Graph(id='graph4-page1')
        ])
    ])
])

# Callback pour récupérer les commentaires, mettre à jour les graphiques et afficher le message de statut
@callback(
    [Output('status-message', 'children'),
     Output('video-info-content', 'children'),
     Output('graph1-page1', 'figure'),
     Output('graph2-page1', 'figure'),
     Output('graph3-page1', 'figure'),
     Output('graph4-page1', 'figure')],
    [Input('retrieve-comments-button', 'n_clicks')],
    [Input('video-id-input', 'value')]
)
def update_graphs(n_clicks, video_id):
    if n_clicks > 0:
        if not video_id or len(video_id) != 11:
            return "Invalid Video ID. It must be 11 characters long.", html.Div(), {}, {}, {}
        
        start_time = time.time()
        status_message = "Retrieving comments and video info..."

        # Récupérer les commentaires et les infos de la vidéo
        # try:
        if not os.path.exists(f"data/infos/{video_id}_infos.json") or not os.path.exists(f"data/comments/{video_id}.txt"):
            commentaires, infos_video = obtenir_infos_video_et_commentaires(video_id, API_KEY)
            status_message = "Comments and video info retrieved successfully."
        else:
            # Charger les commentaires et les infos de la vidéo depuis les fichiers
            with open(f"data/comments/{video_id}.txt", "r", encoding="utf-8") as f:
                commentaires = [line.strip() for line in f.readlines()]
            with open(f"data/infos/{video_id}_infos.json", "r", encoding="utf-8") as json_file:
                infos_video = json.load(json_file)
            status_message = "Comments and video info loaded from file."

        commentaires = [c for c in commentaires if c]  # Filter out empty strings and None values
        df_commentaires = pd.DataFrame([{'comment': c} for c in commentaires])
        df_commentaires = df_commentaires.dropna(subset=['comment'])
        # print(df_commentaires)
        
        # Analyser les commentaires
        if not os.path.exists(f"data/dataframes/{video_id}.csv"):
            status_message = "Analyzing comments..."
            df_commentaires = comment_analysis(df_commentaires, nlp, video_id)
        else:
            df_commentaires = pd.read_csv(f"data/dataframes/{video_id}.csv")
            status_message = "Comment analysis loaded from file."

        # Générer les graphiques
        status_message = "Generating graphs..."
        wordcloud_fig = word_cloud(df_commentaires,nlp)
        polarity_fig = polarity_plot(df_commentaires)
        subjectivity_fig = subjectivity_plot(df_commentaires)
        gauge_polarity_fig, nb_comments = polarity_on_vpn(df_commentaires)

        end_time = time.time()
        elapsed_time = end_time - start_time
        status_message = f"Analysis completed in {elapsed_time:.2f} seconds."

        # Créer le contenu des informations de la vidéo
        video_info_content = html.Div([
            html.P(f"Titre : {infos_video['titre_video']}"),
            html.P(f"Description : {infos_video['description_video']}"),
            html.P(f"Date de publication : {infos_video['date_publication']}"),
            html.P(f"Canal : {infos_video['canal']}"),
            html.P(f"Nombre de vues : {infos_video['vue_count']}"),
            html.P(f"Nombre de likes : {infos_video['like_count']}"),
            html.P(f"Nombre de commentaires : {infos_video['comment_count']}"),
            html.P(f"Nombre de commentaires VPN : {nb_comments}"),
        ])
        
        return status_message, video_info_content, wordcloud_fig, polarity_fig, subjectivity_fig, gauge_polarity_fig

        # except Exception as e:
        #     return f"An error occurred: {str(e)}", html.Div(), {}, {}, {}, {}

    # Valeurs par défaut si aucun commentaire n'est récupéré
    return "", html.Div(), {}, {}, {}, {}
