import json
import pandas as pd
from googleapiclient.discovery import build
    
def obtenir_infos_video_et_commentaires(video_id, api_key):
    youtube = build("youtube", "v3", developerKey=api_key)
    
    # Obtenir les informations de la vidéo
    video_request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    video_response = video_request.execute()
    
    if video_response["items"]:
        video_info = video_response["items"][0]["snippet"]
        video_stats = video_response["items"][0]["statistics"]
        
        # Extraire les informations pertinentes
        infos_video = {
            "titre_video": video_info["title"],
            "description_video": video_info["description"],
            "date_publication": video_info["publishedAt"],
            "canal": video_info["channelTitle"],
            "vue_count": video_stats.get("viewCount", 0),
            "like_count": video_stats.get("likeCount", 0),
            "comment_count": video_stats.get("commentCount", 0)
        }
    else:
        infos_video = {
            "titre_video": "Titre inconnu",
            "description_video": "Description inconnue",
            "date_publication": "Date inconnue",
            "canal": "Canal inconnu",
            "vue_count": 0,
            "like_count": 0,
            "comment_count": 0
        }
    
    # Obtenir les commentaires
    commentaires = []
    requete = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=100
    )
    reponse = requete.execute()

    while reponse:
        for item in reponse["items"]:
            commentaire = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            commentaires.append(commentaire)

        # Pagination
        if "nextPageToken" in reponse:
            requete = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                pageToken=reponse["nextPageToken"],
                maxResults=100
            )
            reponse = requete.execute()
        else:
            break
        
    # enregistrement des infos dans un fichier
    with open(f"data/infos/{video_id}_infos.json", "w", encoding="utf-8") as json_file:
        json.dump(infos_video, json_file, ensure_ascii=False, indent=4)
    
    # enregistrement des commentaires dans un fichier
    with open(f"data/comments/{video_id}.txt", "w", encoding="utf-8") as f:
        for commentaire in commentaires:
            f.write(commentaire + "\n")

    return commentaires, infos_video

