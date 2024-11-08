from dash import html, dcc, register_page, Input, Output, callback
from utils.def_api_google import obtenir_infos_video_et_commentaires
import os
import pandas as pd
import time

API_KEY = os.getenv("API_KEY")
register_page(__name__, path="/page1")

layout = html.Div([ ])

