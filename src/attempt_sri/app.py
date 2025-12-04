# app.py
import streamlit as st
import os
import json
import pandas as pd
from pathlib import Path

# ===================== CONFIGURATION STREAMLIT =====================
st.set_page_config(
    page_title="CineFinder ",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# CineFinder PRO\nMoteur de recherche intelligent basé sur BM25 + WordNet + Lemmatisation contextuelle"
    }
)

# ===================== CHARGEMENT DES DONNÉES =====================
@st.cache_data
def load_movies_data(docs_dir: str):
    movies = []
    path = Path(docs_dir)
    for file in path.glob("row_*.json"):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data["doc_id"] = file.stem  # row_1, row_2, etc.
                movies.append(data)
        except Exception as e:
            continue
    return pd.DataFrame(movies)

# Chemin vers tes fichiers
DOCS_DIR = Path(r"D:\GI5\BOUZID\Projet\CineFinder\data\docs")
movies_df = load_movies_data(DOCS_DIR)

# ===================== INITIALISATION DU MOTEUR =====================
@st.cache_resource
def init_engine():
    from search_engine import MovieSearchEngine
    engine = MovieSearchEngine()
    engine.index_documents(str(DOCS_DIR))
    return engine

with st.spinner("Initialisation du moteur CineFinder PRO (BM25 + WordNet)..."):
    engine = init_engine()

# ===================== FONCTIONS D'AFFICHAGE =====================
def display_poster(poster_path, width=200):
    if poster_path and pd.notna(poster_path):
        url = f"https://image.tmdb.org/t/p/w300{poster_path}"
        return st.image(url, width=width)
    return st.image("https://via.placeholder.com/300x450.png?text=No+Poster", width=width)

def get_bm25_badge(score):
    norm_score = min(score / 20.0, 1.0)  # Normalisation approximative (BM25 max ≈ 15-25)
    if norm_score >= 0.7:
        color, label = "#0d6efd", "Excellent"
    elif norm_score >= 0.5:
        color, label = "#198754", "Très bon"
    elif norm_score >= 0.3:
        color, label = "#ffc107", "Bon"
    else:
        color, label = "#dc3545", "Faible"
    return f"""
    <span style="
        background:{color}; color:white; padding:6px 14px; border-radius:20px;
        font-weight:bold; font-size:13px; display:inline-block;
    ">{label} • {score:.2f}</span>
    """

# ===================== INTERFACE =====================
st.title("🎬 CineFinder PRO")
st.markdown("### Le moteur de recherche de films **le plus intelligent** jamais créé en GI5")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "🔍 Décris le film que tu cherches",
        placeholder="ex: dark knight nolan, bateau qui coule, robot comique, rêve dans le rêve...",
        label_visibility="collapsed"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Rechercher", type="primary", use_container_width=True):
        pass

# Filtres latéraux
with st.sidebar:
    st.header("🎛️ Filtres")
    genres_list = ["Tous les genres"]
    for genres in movies_df["Genres"].dropna():
        genres_list.extend([g.strip() for g in str(genres).split(",")])
    selected_genre = st.selectbox("Genre", sorted(set(genres_list)))

    years = ["Toutes les années"] + sorted(
        movies_df["Release_Date"].str[:4].dropna().unique(), reverse=True
    )
    selected_year = st.selectbox("Année", years)

    min_rating = st.slider("Note minimale (TMDB)", 0.0, 10.0, 6.0, step=0.5)

    st.markdown("---")
    st.markdown(f"**Collection chargée** : {len(movies_df)} films")
    st.markdown("**Ranking** : BM25 (standard industriel)")
    st.markdown("**NLP** : Lemmatisation + POS + WordNet")
# ===================== RECHERCHE =====================
if query.strip():
    with st.spinner("Recherche en cours avec BM25 + expansion synonymique..."):
        results = engine.search(query, top_k=50, debug=False)

    if not results:
        st.warning("Aucun résultat trouvé. Essaie une autre formulation !")
        st.stop()

    # Filtrage post-recherche
    filtered = []
    for doc_id, score, meta in results:
        row = movies_df[movies_df["doc_id"] == doc_id].iloc[0]

        if selected_genre != "Tous les genres" and selected_genre not in str(row.get("Genres", "")):
            continue
        if selected_year != "Toutes les années" and row.get("Release_Date", "")[:4] != selected_year:
            continue
        if float(row.get("Vote_Average", 0)) < min_rating:
            continue

        filtered.append((score, row))

    st.success(f"**{len(filtered)} film(s)** trouvé(s) pour « **{query}** »")

    # Affichage des résultats
    for i, (score, film) in enumerate(filtered[:20], 1):
        with st.container():
            col1, col2 = st.columns([1, 4])

            with col1:
                display_poster(film.get("Poster_Path"))

            with col2:
                st.markdown(f"### {i}. **{film['Title']}** ({film['Release_Date'][:4]})")
                st.markdown(get_bm25_badge(score), unsafe_allow_html=True)

                director = film.get("Director", "Inconnu")
                genres = film.get("Genres", "")
                rating = film.get("Vote_Average", 0)

                st.caption(f"**Réalisateur** : {director} | **Genres** : {genres}")
                st.progress(min(score / 25.0, 1.0))
                st.write(f"⭐ **Note TMDB** : {rating:.1f}/10 | **Score BM25** : {score:.3f}")

                with st.expander("Voir le synopsis"):
                    st.write(film.get("Overview", "Pas de résumé disponible."))

            st.markdown("---")

else:
    st.info("Entrez une recherche pour commencer !")
    st.markdown("""
    #### Exemples de requêtes qui marchent parfaitement :
    - `dark knight nolan`
    - `bateau qui coule`
    - `rêve dans le rêve`
    - `christopher nolan`
    - `film triste guerre`
    - `robot comique pixar`
    - `quentin tarantino western`
    """)

# Footer
st.markdown(
    "<div style='text-align: center; padding: 20px; color: #666; font-size: 14px;'>"
    "CineFinder  • BM25 + WordNet + Lemmatisation contextuelle "
    "</div>",
    unsafe_allow_html=True
)