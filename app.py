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

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Importer le moteur sémantique
from src.search_engine import search_documents

# Charger le dataset complet (affiches + détails)
@st.cache_data
def load_full_data():
    return pd.read_csv("data/cleaned_movies.csv")

data_full = load_full_data()

# Fonction d'affichage d'affiche
def safe_poster(poster_path, width=100):
    try:
        if pd.notna(poster_path) and str(poster_path).strip():
            return st.image(f"https://image.tmdb.org/t/p/w154{poster_path}", width=width)
    except:
        pass
    return st.image("https://via.placeholder.com/100x150.png?text=?", width=width)

# Couleur selon score sémantique
def get_similarity_color(score):
    if score >= 0.7:
        return "#28a745"
    elif score >= 0.4:
        return "#ffc107"
    else:
        return "#dc3545"

# Badge
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

# ====================== INTERFACE ======================
st.title("🎬 CinéFinder")
st.markdown("### Le moteur de recherche qui **comprend le sens** de ta requête")

# Barre de recherche
query = st.text_input(
    "Décris le film que tu cherches",
    placeholder="ex: bateau qui coule, voyage dans l’espace, film romantique triste..."
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
    years = ["Toutes"] + sorted(
        data_full["Release_Date"].astype(str).str[:4].dropna().unique(),
        reverse=True
    )
    selected_year = st.selectbox("Année", years)
    year_filter = None if selected_year == "Toutes" else selected_year

with col3:
    min_rating = st.slider("Note minimale (TMDB)", 0.0, 10.0, 0.0, step=0.5)


# ==================== SEARCH ====================
if st.button("🔍 Rechercher", type="primary") or query:

    if query.strip():

        with st.spinner("Recherche sémantique en cours..."):

            results = search_documents(
                query=query,
                genre_filter=genre_filter,
                year_filter=year_filter
            )

            # Filtrer par note minimale
            if min_rating > 0:
                results = [
                    r for r in results
                    if float(r.get("Rating", 0) or 0) >= min_rating
                ]

        st.success(f"✅ {len(results)} résultat(s) trouvé(s) pour « **{query}** »")

        if len(results) > 0:

            df_results = pd.DataFrame(results)

            avg_similarity = df_results["Similarity"].mean()
            max_similarity = df_results["Similarity"].max()

            col_stat1, col_stat2, col_stat3 = st.columns(3)

            with col_stat1:
                st.metric("Meilleur match", f"{int(max_similarity * 100)}%")
            with col_stat2:
                st.metric("Similarité moyenne", f"{int(avg_similarity * 100)}%")
            with col_stat3:
                st.metric("Résultats affichés", len(results))

            st.markdown("---")

            # AFFICHAGE
            for idx, film_data in enumerate(results, 1):

                match = data_full[
                    (data_full["Title"] == film_data["Title"]) &
                    (data_full["Release_Date"].astype(str).str[:4] == film_data["Year"])
                ]

                if not match.empty:
                    film = match.iloc[0]
                else:
                    film = film_data

                col1, col2 = st.columns([1, 5])

                with col1:
                    safe_poster(film.get("Poster_Path"))

                with col2:

                    title_col, badge_col = st.columns([4, 1])

                    with title_col:
                        st.markdown(f"### **{idx}. {film.get('Title')}** ({film.get('Release_Date', '')[:4]})")

                    with badge_col:
                        st.markdown(
                            similarity_badge(film_data["Similarity"]),
                            unsafe_allow_html=True
                        )

                    st.caption(f"🎥 **{film.get('Director','Inconnu')}** • {film.get('Genres','')}")

                    colA, colB = st.columns(2)

                    with colA:
                        st.write(f"⭐ **Note TMDB:** {film.get('Vote_Average', 0):.1f}/10")

                    with colB:
                        st.write(f"🧠 **Pertinence:** {film_data['Similarity']:.3f}")

                    st.progress(
                        film_data["Similarity"],
                        text=f"Score sémantique : {int(film_data['Similarity'] * 100)}%"
                    )

                    if "Overview" in film and pd.notna(film["Overview"]):
                        with st.expander("📖 Voir le résumé"):
                            st.write(film["Overview"])

                st.markdown("---")

        else:
            st.info("❌ Aucun résultat pertinent trouvé. Essaie une autre formulation.")

    else:
        st.warning("⚠️ Entre une requête pour commencer.")


# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("""
    🧠 **Recherche sémantique (embeddings)**

    - 🟢 70–100% : Excellente correspondance
    - 🟡 40–69% : Moyenne
    - 🔴 < 40% : Faible
    """)

    st.markdown("---")
    st.markdown(f"🎬 **Base de données :** {len(data_full)} films")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Propulsé par des embeddings (Sentence-BERT) 🚀</div>",
    unsafe_allow_html=True
)
