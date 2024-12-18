import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import plotly.graph_objects as go
from collections import Counter
import pandas as pd
import os
import nltk
from transformers import pipeline
import spacy

nlp = spacy.load("fr_core_news_sm")

nltk.download('punkt_tab')
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
def get_polarity(comment):    
    tokenized_comment = nltk.sent_tokenize(comment[:512])
    vs = classifier(tokenized_comment)
    number = int(vs[0]['label'].split()[0]) 
    
    return number

def comment_analysis(df, video_id):
    df['Number'] = df['Comment'].apply(lambda x: get_polarity(x))
    
     # enregistrement du df dans un fichier
    df.to_csv(f"data/dataframes/{video_id}.csv", index=False)
    
    return df

# Nuage de mots
def word_cloud(df, nlp, keyword_sheet_id=None, want_to_save=False):
    if os.path.exists(f"data/keywords/{keyword_sheet_id}.csv"):
        keywords_df = pd.read_csv(f"data/keywords/{keyword_sheet_id}.csv")
        counter = Counter(dict(zip(keywords_df['keyword'], keywords_df['count'])))
    else:
        df = df[df['Comment'].str.contains(r'vpn|nord', case=False, na=False)]

        keywords = []
        for comment in df['Comment']:
            doc = nlp(comment)
            for token in doc:
                if token.is_alpha and len(token.text) > 2:
                    keywords.append(token.lemma_.lower())

        keywords = [keyword for keyword in keywords if keyword not in ['vpn', 'nord', 'nordvpn', 'nvpn']]
        keywords = [keyword for keyword in keywords if keyword not in nlp.Defaults.stop_words]
        
        counter = Counter(keywords)
        if want_to_save:
            pd.DataFrame(counter.items(), columns=['keyword', 'count']).sort_values('count', ascending=False).to_csv(f"data/keywords/{keyword_sheet_id}.csv", index=False)


    wordcloud = WordCloud(width=3000, height=1000, background_color='white', stopwords=None, min_font_size=10).generate_from_frequencies(counter)

    fig = px.imshow(wordcloud, title="Nuage de mots des Commentaires")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig


#Graphique de polarité
def polarity_plot(df):
    fig = px.pie(df, names=['Positif', 'Neutre', 'Négatif'], 
                    values=[(df['Number'] > 3).sum(), (df['Number'] == 3).sum(), (df['Number'] < 3).sum()],
                    title="Répartition des Sentiments des Commentaires")
    return fig

def polarity_on_vpn(df):
    df_vpn = df[df['Comment'].str.contains(r'vpn|nord|sponso', case=False, na=False)]
    polarity_vpn = df_vpn['Number'].mean()
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=polarity_vpn,
        title={"text": "Sentiment Moyen des Commentaires Contenant 'VPN'"},
        gauge={
            'axis': {'range': [1, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [1, 3], 'color': "red"},
                {'range': [3, 5], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': polarity_vpn
            }
        }
    ))
    fig.update_layout(
        title_font_size=24,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )
    return fig, len(df_vpn)

#Graphique de subjectivité
def subjectivity_plot(df):
    fig = px.pie(df, names=['Subjectif', 'Objectif'], 
                    values=[(df['subjectivity'] > 0.5).sum(), (df['subjectivity'] <= 0.5).sum()],
                    title="Répartition des Commentaires Subjectifs et Objectifs")
    return fig


# Dataframe de tout les commentaires
def all_comments():
    df_commentaires =  pd.DataFrame({})
    for file in os.listdir("data/dataframes"):
        df = pd.read_csv(f"data/dataframes/{file}")
        df['video_id'] = file.split(".")[0]        
        infos = pd.read_json(f'data/infos/{file.split(".")[0]}_infos.json', typ='series')
        df['date_publication'] = infos['date_publication']
        df['youtubeur'] = infos['canal']
        df_commentaires = pd.concat([df_commentaires, df])
    return df_commentaires

# Evolution de la polarité des commentaires d'un youtubeur
def polarity_evolution(df):
    df_tmp = df[df['Comment'].str.contains(r'vpn|nord|sponso', case=False, na=False)]

    df_tmp['date_publication'] = pd.to_datetime(df_tmp['date_publication'])
    
    polarity_by_date = df_tmp.groupby(df_tmp['date_publication'].dt.date)['Number'].mean()
    comments_by_date = df_tmp.groupby(df_tmp['date_publication'].dt.date).size()
    
    fig = go.Figure()
    # Trace pour la polarité moyenne
    fig.add_trace(go.Scatter(x=polarity_by_date.index, y=polarity_by_date.values,
                             mode='lines+markers', name='Mean Polarity', yaxis='y1'))
    # Trace pour le nombre de commentaires
    fig.add_trace(go.Bar(x=comments_by_date.index, y=comments_by_date.values,
                         name='Number of Comments', opacity=0.5, yaxis='y2'))
    # Trace pour la polarité neutre
    fig.add_trace(go.Scatter(x=polarity_by_date.index, y=[3]*len(polarity_by_date),
                             mode='lines', name='Neutral Polarity', yaxis='y1', line=dict(dash='dash', color='gray')))

    fig.update_layout(
        title='Evolution de la Polarité des Commentaires parlant de NordVPN',
        xaxis_title='Date de Publication de la vidéo',
        yaxis=dict(title='Polarity Moyenne', side='left', showgrid=False, range=[1, 5]),
        yaxis2=dict(title='Nombre de commentaires', side='right', overlaying='y', showgrid=False),
        barmode='overlay'
    )
    return fig

def polarity_evolution2(df):
    df_tmp = df.copy()
    df_tmp['date_publication'] = pd.to_datetime(df_tmp['date_publication'])
    
    polarity_by_date = df_tmp.groupby(df_tmp['date_publication'].dt.date)['Number'].mean()
    comments_by_date = df_tmp.groupby(df_tmp['date_publication'].dt.date).size()
    
    fig = go.Figure()
    # Trace pour la polarité moyenne
    fig.add_trace(go.Scatter(x=polarity_by_date.index, y=polarity_by_date.values,
                             mode='lines+markers', name='Mean Polarity', yaxis='y1'))
    # Trace pour le nombre de commentaires
    fig.add_trace(go.Bar(x=comments_by_date.index, y=comments_by_date.values,
                         name='Number of Comments', opacity=0.5, yaxis='y2'))

    # Trace pour la polarité neutre
    fig.add_trace(go.Scatter(x=polarity_by_date.index, y=[3]*len(polarity_by_date),
                             mode='lines', name='Neutral Polarity', yaxis='y1', line=dict(dash='dash', color='gray')))
    
    fig.update_layout(
        title='Evolution de la Polarité de tous les Commentaires',
        xaxis_title='Date de Publication de la vidéo',
        yaxis=dict(title='Polarity Moyenne', side='left', showgrid=False, range=[1, 5]),
        yaxis2=dict(title='Nombre de commentaires', side='right', overlaying='y', showgrid=False),
        barmode='overlay'
    )
    return fig

# Comparaison des youtubeurs
def compare_youtubeurs(df, youtubeur):
    df_tmp = df[df['Comment'].str.contains(r'vpn|nord|sponso', case=False, na=False)]

    polarity_by_youtubeur = df_tmp.groupby(df_tmp['youtubeur'])['Number'].mean().sort_values(ascending=False)
    
    colors = ['red' if yt == youtubeur else 'skyblue' for yt in polarity_by_youtubeur.index]
    
    fig = go.Figure(go.Bar(
        x=polarity_by_youtubeur.values,
        y=polarity_by_youtubeur.index,
        orientation='h',
        marker=dict(color=colors)
    ))
    
    # Ajouter une barre verticale à x=3
    fig.add_shape(
        type="line",
        x0=3,
        y0=0,
        x1=3,
        y1=1,
        xref='x',
        yref='paper',
        line=dict(color="black", width=2, dash="dash")
    )
    
    fig.update_layout(
        title='Comparaison de la Polarité des Commentaires parlant de VPN entre les YouTubers',
        xaxis_title='Mean Polarity',
        xaxis=dict(range=[1, 5]),
        yaxis_title='YouTubers',
        yaxis=dict(autorange='reversed')  # Pour inverser l'ordre des YouTubers
    )
    return fig

def compare_youtubeurs2(df, youtubeur):
    # Dataframe avec par youtubeur et par video_id le nombre de commentaires total et le nombre de commentaires parlant de vpn
    comments_by_youtubeur = df.groupby(['youtubeur', 'video_id']).size().reset_index(name='count')
    df_tmp = df[df['Comment'].str.contains(r'vpn|nord|sponso', case=False, na=False)]

    vpn_comments_by_youtubeur = df_tmp.groupby(['youtubeur', 'video_id']).size().reset_index(name='count')

    # merge des deux dataframes
    df_youtubeurs = pd.merge(comments_by_youtubeur, vpn_comments_by_youtubeur, on=['youtubeur', 'video_id'], suffixes=('_total', '_vpn'))
    df_youtubeurs['vpn_part'] = df_youtubeurs['count_vpn'] / df_youtubeurs['count_total']
    
    # Nombre moyen de commentaires vpn par youtubeur    
    nb_comments_by_youtubeur = df_youtubeurs.groupby('youtubeur')['count_vpn'].mean().sort_values(ascending=False)
        
    # Part moyenne des commentaires parlant de vpn par youtubeur
    part_comments_by_youtubeur = df_youtubeurs.groupby('youtubeur')['vpn_part'].mean()
    
    # Réalignement pour le scatter plot
    part_comments_by_youtubeur = part_comments_by_youtubeur[nb_comments_by_youtubeur.index]
    
    fig = go.Figure()
    # Trace pour le nombre moyen de commentaires vpn
    colors_nb = ['red' if yt == youtubeur else 'skyblue' for yt in nb_comments_by_youtubeur.index]
    fig.add_trace(go.Bar(x=nb_comments_by_youtubeur.index, y=nb_comments_by_youtubeur.values,
                            name='Mean Number of VPN Comments', yaxis='y1', marker_color=colors_nb))
    
    # Trace pour la part moyenne des commentaires vpn
    fig.add_trace(go.Scatter(x=nb_comments_by_youtubeur.index, y=part_comments_by_youtubeur.values,
                                mode='lines+markers', name='Mean Part of VPN Comments', yaxis='y2', 
                                line=dict(color='orange'), connectgaps=False))
    
    fig.update_layout(
        title='Comparaison de la quantité de Commentaire parlant de VPN entre les YouTubers',
        xaxis_title='YouTubers',
        yaxis=dict(title='Mean Number of VPN Comments', side='left', showgrid=False),
        yaxis2=dict(title='Mean Part of VPN Comments', side='right', overlaying='y', showgrid=False),
        barmode='overlay'
    )
    return fig

def bigramme(df):
    # Filtrer les commentaires contenant des mentions relatives à VPN
    keywords = ['vpn', 'nordvpn', 'nvpn', 'nord vpn']
    df = df[df['Comment'].str.contains('|'.join(keywords), case=False, na=False)]
    
    bigram = Counter()
    
    for comment in df['Comment']:
        # Traiter chaque commentaire avec spaCy
        doc = nlp(comment.lower())
        words = [
            token.lemma_ for token in doc
            if token.is_alpha and not token.is_stop  # Conserver seulement les mots significatifs
        ]
        
        # Construire les bigrammes
        for i in range(len(words) - 1):
            bigram[(words[i], words[i + 1])] += 1
    
    # Trier et sélectionner les 20 bigrammes les plus fréquents
    bigram = bigram.most_common(20)
    
    # Convertir en DataFrame pour une visualisation facile
    bigram_df = pd.DataFrame(bigram, columns=['bigram', 'count'])
    bigram_df['bigram'] = bigram_df['bigram'].apply(lambda x: ' '.join(x))  # Joindre les bigrammes en texte
    
    # Créer un graphique interactif avec Plotly
    fig = px.bar(
        bigram_df,
        x='bigram',
        y='count',
        title="Bigrammes des Commentaires contenant des mots liés à NordVPN",
        labels={'bigram': 'Bigramme', 'count': 'Nombre d\'occurrences'},
        text='count'
    )
    
    # Affiner l'apparence du graphique
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig    
    
def trigramme(df):
    # Filtrer les commentaires contenant des mentions relatives à VPN
    keywords = ['vpn', 'nordvpn', 'nvpn', 'nord vpn']
    df = df[df['Comment'].str.contains('|'.join(keywords), case=False, na=False)]
    
    trigram = Counter()
    
    for comment in df['Comment']:
        # Traiter chaque commentaire avec spaCy
        doc = nlp(comment.lower())
        words = [
            token.lemma_ for token in doc
            if token.is_alpha and not token.is_stop  # Conserver seulement les mots significatifs
        ]
        
        # Construire les trigrammes
        for i in range(len(words) - 2):  # Adapté pour trigrammes
            trigram[(words[i], words[i + 1], words[i + 2])] += 1
    
    # Trier et sélectionner les 20 trigrammes les plus fréquents
    trigram = trigram.most_common(20)
    
    # Convertir en DataFrame pour une visualisation facile
    trigram_df = pd.DataFrame(trigram, columns=['trigram', 'count'])
    trigram_df['trigram'] = trigram_df['trigram'].apply(lambda x: ' '.join(x))  # Joindre les trigrammes en texte
    
    # Créer un graphique interactif avec Plotly
    fig = px.bar(
        trigram_df,
        x='trigram',
        y='count',
        title="Trigrammes des Commentaires contenant des mots liés à NordVPN",
        labels={'trigram': 'Trigramme', 'count': 'Nombre d\'occurrences'},
        text='count'
    )
    
    # Affiner l'apparence du graphique
    fig.update_layout(xaxis_tickangle=-45)
    
    return fig