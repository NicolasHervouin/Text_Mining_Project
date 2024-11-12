# Projet d'Analyse de Sentiment des Commentaires YouTube

Ce projet utilise des modèles de traitement du langage naturel pour analyser les sentiments et les mots importants des commentaires YouTube.

## Prérequis

1. Créez un environnement virtuel et activez-le :

   python -m venv env

   - **Pour activer l'environnement** :
     - Sur macOS/Linux :
       
       source env/bin/activate
     
     - Sur Windows :
       
       .\.venv\Scripts\activate

2. Installez les dépendances :

   pip install -r requirements.txt

## Configuration des Variables d'Environnement

Certaines variables d’environnement sont nécessaires pour faire fonctionner ce projet (par exemple, des clés API). Suivez les étapes ci-dessous pour les configurer.

1. **Créer un fichier `.env`** : Dans le répertoire racine du projet, créez un fichier nommé `.env` et ajoutez-y vos variables d'environnement sous le format suivant :

   API_KEY="your_api_key_here"  

2. **Charger les variables d'environnement depuis le terminal**.

   - **Sur macOS/Linux** :  
     Ouvrez votre terminal dans le répertoire du projet et exécutez la commande suivante :

     "export $(grep -v '^#' .env | xargs)"

   - **Sur Windows (PowerShell)** :  
     Dans PowerShell, exécutez la commande suivante :

     "Get-Content .env | ForEach-Object {  
         $name, $value = $_ -split '=', 2  
         [System.Environment]::SetEnvironmentVariable($name, $value)  
     }""


## Exécution du Projet

Après avoir chargé les variables d'environnement, vous pouvez exécuter le script principal :

   python app.py

## Fonctionnalités

- **Analyse de Sentiment** : Évalue les commentaires pour en déterminer la polarité (positif, négatif ou neutre).
- **Extraction de Mots Importants** : Identifie les mots et expressions les plus fréquents ou significatifs dans les commentaires.
