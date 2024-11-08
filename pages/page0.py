from dash import html, register_page

register_page(__name__, "/")

layout = html.Div(
    [
        html.Div(
            [
                html.H2("Introduction", style={"text-align": "justify"}),
                html.P(
                    """
                    Ce dashboard a pour objectif de présenter les différentes fonctionnalités de l'application.
                    """,
                    style={"text-align": "justify"}
                ),
            ],
            style={"padding": "20px", "border": "1px solid #ccc", "border-radius": "10px", "box-shadow": "2px 2px 5px rgba(0,0,0,0.1)"}
        )
    ]
)