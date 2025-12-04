# src/app.py  ← remplace tout le fichier par ÇA
import streamlit as st
from data_loader import load_all_films
from vector_db import get_or_create_collection, populate_collection_if_empty

st.set_page_config(page_title="CineFinder", layout="centered")
st.title("CineFinder – Recherche sémantique")
st.caption("50 films • Modèle léger • Zéro conflit • Marche direct sur Mac")

# Chargement des films
with st.spinner("Chargement des 50 films..."):
    films = load_all_films()

if len(films) == 0:
    st.error("Aucun film trouvé dans movies_raw/ !")
    st.stop()

st.success(f"{len(films)} films chargés")

# Base vectorielle
collection = get_or_create_collection()
populate_collection_if_empty(collection, films)

# ←←←← LA LIGNE QUI MANQUAIT ←←←←
query = st.text_input(
    "Décris le film que tu cherches",
    placeholder="ex: film où on entre dans les rêves pour voler une idée",
    key="query"                     # ← cette ligne fait tout marcher
)

# ←←←← et le if juste en dessous
if query:
    with st.spinner("Recherche sémantique..."):
        results = collection.query(
            query_texts=[query],
            n_results=10
        )
    
    st.write(f"### {len(results['ids'][0])} résultats trouvés")
    for i, (meta, dist) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
        score = round(1 - dist, 3)
        with st.expander(f"{i+1}. **{meta['title']}** ({meta['year']}) • Score: {score}", expanded=i<3):
            col1, col2 = st.columns([1, 3])
            with col1:
                if meta['poster']:
                    st.image(f"https://image.tmdb.org/t/p/w500{meta['poster']}", use_column_width=True)
            with col2:
                if meta.get('tagline'):
                    st.caption(f"_{meta['tagline']}_")
                st.write(f"**Genres** : {meta['genres']}")
                st.write(f"**Réal** : {meta['director']}")
                st.write(f"**Acteurs** : {meta['cast']}")
                st.write(meta['overview'] or "Pas de synopsis")

# Sidebar
with st.sidebar:
    st.header("Exemples rapides")
    for exemple in [
        "film dans les rêves",
        "DiCaprio vole des idées",
        "comédie française triste",
        "Nolan",
        "film qui fait pleurer",
        "science-fiction philosophique"
    ]:
        if st.button(exemple, use_container_width=True):
            st.session_state.query = exemple
            st.rerun()