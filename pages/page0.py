from dash import html, register_page

register_page(__name__, "/")

layout = html.Div(
    [   
        html.Div(
            [
                
                html.H2("Introduction", style={"text-align": "center"}),
                html.P(
                    """
                    Bienvenue sur notre dashboard d'analyse des commentaires YouTube sponsorisés par NordVPN. Ce tableau de bord a pour objectif d'offrir une vision complète et dynamique des interactions autour des vidéos de YouTubeurs ayant collaboré avec NordVPN. Voici ce que vous pourrez explorer :
                    """,
                    style={"text-align": "justify"}
                ),
            ],
                style={
                    "padding": "20px",
                    "border": "1px solid #ccc",
                    "border-radius": "10px",
                    "box-shadow": "2px 2px 5px rgba(0,0,0,0.1)",
                    "margin-bottom": "20px",
                },
        ),   
            
        html.Div(
            [        
                html.H2("Page 1 : Analyse Générale", style={"text-align": "left"}),
                html.Ul(
                    [
                        html.Li(
                            html.Span([
                                html.B("1. Évolution de la polarité au fil du temps : "),
                                "Découvrez comment les commentaires évoluent en termes de polarité (positive ou négative) pour chaque YouTubeur au fil des dates de publication des vidéos."
                            ])
                        ),
                        html.Li(
                            html.Span([
                                html.B("2. Comparaison entre les YouTubeurs : "),
                                "Comparez les performances globales des YouTubeurs en termes de sentiment général des commentaires laissés sur leurs vidéos."
                            ])
                        ),
                        html.Li(
                            html.Span([
                                html.B("3. Analyse des mots-clés : "),
                                "Identifiez les termes les plus souvent associés à NordVPN dans les commentaires."
                            ])
                        ),
                        html.Li(
                            html.Span([
                                html.B("4. Nuages de mots selon la polarité : "),
                                "Visualisez deux nuages de mots distincts basés sur les polarités positives et négatives."
                            ])
                        ),
                    ],
                    style={"padding-left": "20px"}
                ),
                html.P(
                    """
                    Cette analyse approfondie permet de mieux comprendre l’impact des campagnes de sponsoring NordVPN sur YouTube. Plongez dans les données, explorez les tendances et découvrez les réactions des audiences face à NordVPN !
                    """,
                    style={"text-align": "justify"}
                ),
            ],
            style={
                "padding": "20px",
                "border": "1px solid #ccc",
                "border-radius": "10px",
                "box-shadow": "2px 2px 5px rgba(0,0,0,0.1)",
                "margin-bottom": "20px",
            },
        ),
        html.Div(
            [
                html.H2("Page 2 : Analyse vidéo par vidéo", style={"text-align": "left"}),
                html.P(
                    """
                    Dans cette page, vous pourrez sélectionner une vidéo spécifique et accéder à une analyse détaillée des commentaires qui lui sont associés. Vous aurez également la possibilité d'explorer la polarité des commentaires, les mots-clés récurrents, et bien plus encore.
                    """,
                    style={"text-align": "justify"}
                ),
            ],
            style={
                "padding": "20px",
                "border": "1px solid #ccc",
                "border-radius": "10px",
                "box-shadow": "2px 2px 5px rgba(0,0,0,0.1)",
            },
        ),
    ]
)
