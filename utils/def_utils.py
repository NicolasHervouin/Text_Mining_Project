import re
import plotly.express as px
from spacy.language import Language
from spacy.tokens import Doc
from spacy_langdetect import LanguageDetector
from spacytextblob.spacytextblob import SpacyTextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import plotly.graph_objects as go
from textblob import TextBlob
import emoji 
from collections import Counter
import re
    
@Language.factory("language_detector")
def get_lang_detector(nlp, name):
    return LanguageDetector()

def detect_language_spacy(comment, nlp):
    doc = nlp(comment)
    return doc._.language["language"]

def clean_comment(comment):
    # Supprimer les emojis
    comment = re.sub(r'[^\w\s,]', '', comment, flags=re.UNICODE)
    # Retirer les liens
    comment = re.sub(r'http\S+|www.\S+', '', comment)
    # Retirer les caractères spéciaux et ponctuation
    comment = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', comment)    
    return comment

def get_polarity(comment):
    # Gestion des emojis
    emoji_dict = {e: emoji.demojize(e) for e in comment if e in emoji.EMOJI_DATA}
    
    # Pré-traitement du commentaire
    for e, desc in emoji_dict.items():
        comment = comment.replace(e, desc)

    # Utilisation de TextBlob pour analyser la polarité
    blob = TextBlob(comment)
    polarity = blob.sentiment.polarity

    # Emojis supplémentaires pour influencer la polarité
    positive_emojis = [
        ":smiling_face_with_heart_eyes:", ":thumbs_up:", ":red_heart:", ":grinning_face:", 
        ":star_struck:", ":clapping_hands:", ":party_popper:", ":sparkling_heart:",
        ":beaming_face_with_smiling_eyes:", ":face_blowing_a_kiss:", ":winking_face:", ":ok_hand:"
    ]
    negative_emojis = [
        ":angry_face:", ":thumbs_down:", ":crying_face:", ":pouting_face:", 
        ":face_with_symbols_on_mouth:", ":loudly_crying_face:", ":face_with_steam_from_nose:", 
        ":disappointed_face:", ":frowning_face:", ":worried_face:", ":broken_heart:", ":persevering_face:"
    ]

    # Comptage des emojis dans le commentaire
    emoji_counts = Counter(emoji_dict.values())
    
    # Ajustement de la polarité selon les emojis trouvés
    for emj in positive_emojis:
        polarity += 0.1 * emoji_counts[emj]
    for emj in negative_emojis:
        polarity -= 0.1 * emoji_counts[emj]

    return polarity


    

 # Subjectivity donne le score de subjectivité de 0 (objectif) à 1 (subjectif)
def get_subjectivity(comment, nlp):
    doc = nlp(comment)
    return doc._.blob.subjectivity 

# Assessments donne le score de sentiment de -1 (négatif) à 1 (positif)
def get_assessments(comment, nlp):
    doc = nlp(comment)
    return doc._.blob.sentiment_assessments.assessments  

# Extraire les noms et adjectifs 
def extract_keywords(text, nlp):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'ADJ']]
    return keywords 
    
def comment_analysis(df, nlp, video_id):
    df['clean_comment'] = df['comment'].apply(lambda x: clean_comment(x))
    df['language'] = df['comment'].apply(lambda x: detect_language_spacy(x, nlp))
    df['polarity'] = df['comment'].apply(lambda x: get_polarity(x))
    df['subjectivity'] = df['comment'].apply(lambda x: get_subjectivity(x, nlp))
    df['assessments'] = df['comment'].apply(lambda x: get_assessments(x, nlp))
    df['keywords'] = df['comment'].apply(lambda x: extract_keywords(x, nlp))
    
     # enregistrement du df dans un fichier
    df.to_csv(f"data/dataframes/{video_id}.csv", index=False)
    
    return df

#Pos Tagging
def pos_tag(text,nlp):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

#Lemmatisation
def lemmatize(text,nlp):
    doc = nlp(text)
    return [token.lemma_ for token in doc]

# Nuage de mots
def word_cloud(df,nlp):
    df['clean_comment'] = df['clean_comment'].fillna("")

    
    df['POS_Tags'] = df['clean_comment'].apply(lambda x: pos_tag(x, nlp) if isinstance(x, str) and x else None)
    df['Nouns'] = df['POS_Tags'].apply(lambda tags: [tag[0] for tag in tags if tag[1] == 'NOUN'] if tags is not None else [])
    df['Noun_Lemmas'] = df['Nouns'].apply(lambda nouns: lemmatize(' '.join(nouns), nlp))

    keywords = df['Noun_Lemmas'].tolist()
    keywords = [item for sublist in keywords for item in sublist]
    keywords = [word for word in keywords if len(word) > 2]

    counter = Counter(keywords)
    
    wordcloud = WordCloud(width=3000, height=1000, 
                          background_color='white', 
                          stopwords=None, 
                          min_font_size=10).generate_from_frequencies(counter)

    fig = px.imshow(wordcloud, title="Nuage de mots des Commentaires")
    fig.update_xaxes(visible=False) 
    fig.update_yaxes(visible=False)
    
    return fig


#Graphique de polarité
def polarity_plot(df):
    fig = px.pie(df, names=['Positif', 'Neutre', 'Négatif'], 
                    values=[(df['polarity'] > 0).sum(), (df['polarity'] == 0).sum(), (df['polarity'] < 0).sum()],
                    title="Répartition des Sentiments des Commentaires")
    return fig

def polarity_on_vpn(df):
    df_vpn = df[df['comment'].str.contains('vpn', case=False)]
    polarity_vpn = df_vpn['polarity'].mean()
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=polarity_vpn,
        title={"text": "Sentiment Moyen des Commentaires Contenant 'VPN'"},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, 0], 'color': "red"},
                {'range': [0, 1], 'color': "green"}
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