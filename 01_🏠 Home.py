import streamlit as st

# Full size page
st.set_page_config(
    page_title = "ANALYSE - PROFIL FORCE-VITESSE",
    page_icon = "🔬",
    layout="wide")

# Hide Hamburger (Top Right Corner) and "Made with Streamlit" footer:
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.markdown(f'<h1 style="background-color:#6497b1;border-radius:14px;text-align:center;">{"APPLICATION | PROFIL FORCE - VITESSE"}</h1>', unsafe_allow_html=True)
f"*version 2.10 - par Corentin Casali*"
f"""Bienvenue dans l'application profil force-vitesse. Vous trouverez dans le menu à gauche 2 pages, vous permettant de traiter vos fichiers de profil force-vitesse.\n
Vous pouvez utilisez des **fichiers .tdms** ou alors des **fichiers .xls**"""

st.info("""De plus amples informations sont disponibles dans les différentes pages. N'hésitez pas à jeter un coup d'oeil !

Si vous avez des problèmes, merci de contacter : corentin.casali@univ-st-etienne.fr
""", icon="ℹ️")