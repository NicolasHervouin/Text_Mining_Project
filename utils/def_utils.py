import re
import plotly.express as px
from spacy.language import Language
from spacy.tokens import Doc
from spacy_langdetect import LanguageDetector
from spacytextblob.spacytextblob import SpacyTextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
    
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

# Polarity donne le score de sentiment de -1 (négatif) à 1 (positif)
def get_polarity(comment, nlp):
    doc = nlp(comment)
    return doc._.blob.polarity  

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
    df['polarity'] = df['comment'].apply(lambda x: get_polarity(x, nlp))
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
    import ast
    df['POS_Tags'] = df['clean_comment'].apply(lambda x: pos_tag(x, nlp))
    df['Nouns'] = df['POS_Tags'].apply(lambda tags: [tag[0] for tag in tags if tag[1] == 'NOUN'])   
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

#Graphique de subjectivité
def subjectivity_plot(df):
    fig = px.pie(df, names=['Subjectif', 'Objectif'], 
                    values=[(df['subjectivity'] > 0.5).sum(), (df['subjectivity'] <= 0.5).sum()],
                    title="Répartition des Commentaires Subjectifs et Objectifs")
    return fig