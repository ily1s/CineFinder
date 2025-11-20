# app.py
import streamlit as st
import pandas as pd
import sys
import os

st.set_page_config(
    page_title="CinéFinder",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajouter le dossier src au path pour pouvoir importer ton module
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Importer TON moteur existant
from search_engine import search_movies, preprocess_query


# Charger le dataset complet (celui avec les affiches, etc.)
@st.cache_data
def load_full_data():
    return pd.read_csv("data/cleaned_movies.csv")


data_full = load_full_data()


# Fonction pour afficher une image en toute sécurité
def safe_poster(poster_path, width=100):
    try:
        if pd.notna(poster_path) and str(poster_path).strip():
            return st.image(f"https://image.tmdb.org/t/p/w154{poster_path}", width=width)
    except:
        pass
    return st.image("https://via.placeholder.com/100x150.png?text=?", width=width)


# Fonction pour obtenir la couleur selon le score de similarité
def get_similarity_color(score):
    if score >= 0.7:
        return "#28a745"  # Vert
    elif score >= 0.4:
        return "#ffc107"  # Jaune/Orange
    else:
        return "#dc3545"  # Rouge


# Fonction pour afficher le badge de similarité
def similarity_badge(score):
    color = get_similarity_color(score)
    percentage = int(score * 100)
    return f"""
    <div style="
        background-color: {color};
        color: white;
        padding: 5px 12px;
        border-radius: 15px;
        display: inline-block;
        font-weight: bold;
        font-size: 14px;
    ">
        {percentage}% Match
    </div>
    """


# ====================== INTERFACE STREAMLIT ======================
st.title(" CinéFinder")
st.markdown("### Le moteur de recherche qui comprend vraiment ce que tu veux")

# Barre de recherche
query = st.text_input(
    "Décris le film que tu cherches",
    placeholder="ex: bateau qui coule, nolan espace, comédie new york, film triste guerre..."
)

col1, col2, col3 = st.columns(3)
with col1:
    genre_options = ["Tous"] + sorted(
        set(genre.strip() for genres in data_full["Genres"].dropna()
            for genre in str(genres).split(","))
    )
    selected_genre = st.selectbox("Genre", genre_options)
    genre_filter = None if selected_genre == "Tous" else selected_genre

with col2:
    years = ["Toutes"] + sorted(data_full["Release_Date"].str[:4].dropna().unique(), reverse=True)
    selected_year = st.selectbox("Année", years)
    year_filter = None if selected_year == "Toutes" else selected_year

with col3:
    min_rating = st.slider("Note minimale (TMDB)", 0.0, 10.0, 0.0, step=0.5)

if st.button(" Rechercher", type="primary") or query:
    if query.strip():
        with st.spinner("Recherche en cours..."):
            # Utilise TA fonction existante
            results = search_movies(
                query=query,
                top_n=20,
                genre_filter=genre_filter,
                year_filter=year_filter
            )

            # Filtrer par note minimale
            if min_rating > 0:
                results = results[results.index.map(lambda idx:
                                                    data_full[(data_full["Title"] == results.loc[idx, "Title"])][
                                                        "Vote_Average"].iloc[0]
                                                    if len(data_full[(data_full["Title"] == results.loc[
                                                        idx, "Title"])]) > 0 else 0
                                                    ) >= min_rating]

        st.success(f" {len(results)} résultat(s) trouvé(s) pour « **{query}** »")

        if not results.empty:
            # Afficher les statistiques de recherche
            avg_similarity = results["similarity"].mean()
            max_similarity = results["similarity"].max()

            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Meilleur match", f"{int(max_similarity * 100)}%")
            with col_stat2:
                st.metric("Similarité moyenne", f"{int(avg_similarity * 100)}%")
            with col_stat3:
                st.metric("Résultats affichés", len(results))

            st.markdown("---")

            # Afficher les résultats
            for idx, (_, row) in enumerate(results.iterrows(), 1):
                # Récupérer la ligne complète du film dans cleaned_movies.csv via le titre + année
                match = data_full[
                    (data_full["Title"] == row["Title"]) &
                    (data_full["Release_Date"].str[:4] == row["Release_Date"][:4])
                    ]
                if not match.empty:
                    film = match.iloc[0]
                else:
                    film = row  # fallback

                col1, col2 = st.columns([1, 5])
                with col1:
                    safe_poster(film.get("Poster_Path"))

                with col2:
                    # Titre avec badge de similarité
                    title_col, badge_col = st.columns([4, 1])
                    with title_col:
                        st.markdown(f"### **{idx}. {film['Title']}** ({film['Release_Date'][:4]})")
                    with badge_col:
                        st.markdown(similarity_badge(row["similarity"]), unsafe_allow_html=True)

                    st.caption(f" **{film.get('Director', 'Inconnu')}** • {film.get('Genres', '')}")

                    # Affichage de la note et de la similarité
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.write(f"⭐ **Note TMDB:** {film.get('Vote_Average', 0):.1f}/10")
                    with metric_col2:
                        st.write(f" **Pertinence:** {row['similarity']:.3f}")

                    # Barre de progression pour la similarité
                    st.progress(row["similarity"], text=f"Score de correspondance: {int(row['similarity'] * 100)}%")

                    # Résumé
                    if 'Overview' in film and pd.notna(film['Overview']):
                        with st.expander(" Voir le résumé"):
                            st.write(film['Overview'])

                st.markdown("---")
        else:
            st.info(" Aucun résultat... Essaie une autre formulation !")
    else:
        st.warning(" Entre une recherche pour commencer")

with st.sidebar:
    st.markdown("""
     **Score de similarité :**
    - 🟢 **70-100%** : Excellente correspondance
    - 🟡 **40-69%** : Correspondance moyenne
    - 🔴 **0-39%** : Faible correspondance

    """)

    st.markdown("---")
    st.markdown("🎬 **Base de données:** " + str(len(data_full)) + " films")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Propulsé par TF-IDF & Cosine Similarity 🚀</div>",
    unsafe_allow_html=True
)