import streamlit as st
import pandas as pd
import json

st.set_page_config(
    page_title="CineFinder ",
    page_icon="movie_camera",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_bm25_search():
    from src.classification_search.smart_search_loader import run_search
    return run_search

@st.cache_resource
def load_semantic_engine():
    from src.semantic_search.search_engine import search_documents
    return search_documents

@st.cache_data
def load_metadata():
    return pd.read_csv("data/cleaned_movies.csv")

with st.spinner("Chargement des moteurs..."):
    run_search = load_bm25_search()
    semantic_search = load_semantic_engine()
    df = load_metadata()

st.markdown("""
<style>
    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        
    }


    [data-testid="stImage"] img {
        height: 500px !important; 
        width: auto !important;     
        object-fit: contain;     
        border-radius: 8px;
        margin-left: auto;
        margin-right: auto;
        display: block;
        max-width: 160px;          
        text-align: center;
    }
    
    div[data-testid="stMarkdownContainer"] h4 {
        margin-top: 0;
        margin-bottom: 5px;
        font-size: 1.1rem !important;
        text-align: center;
        color: #0d6efd;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis; 
    }

    div[data-testid="stCaptionContainer"] {
        text-align: center;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-color: #333;
        border-radius: 10px;
    }
    .search-container {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #444;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    
    .stTextInput, .stSelectbox {
        margin-bottom: 0px;
    }
    
    div[data-testid="stButton"] {
        padding-top: 26px; 
    }
    
    div[data-testid="stButton"] button {
        border-radius: 8px;
        height: 46px; 
        border: none;
        transition: 0.3s;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h1 style='margin-bottom: 10px;'>CineFinder </h1>
        <h3 style='font-weight: normal; margin-top: 0; color: #aaa;'>
            Le moteur de recherche de films qui cherche le film parfait pour toi !
        </h3>
    </div>
""", unsafe_allow_html=True)


col_spacer_left, col_main, col_spacer_right = st.columns([1, 2, 1])

with col_main:
    with st.container():
        query = st.text_input(
            "Décris le film que tu cherches",
            placeholder=" ex: nolan 2010, tarantino western, bateau qui coule...",
            label_visibility="collapsed"
        )


        sub_c1, sub_c2 = st.columns([3, 1]) 
        
        with sub_c1:
            engine_choice = st.selectbox(
                "Choisis ton moteur",
                ["BM25 + Classification automatique",
                 "Sémantique BERT (compréhension du sens)"],
                label_visibility="visible", 
                help="BM25 → mots-clés exacts / BERT → sens de la phrase"
            )
            
        with sub_c2:
            search = st.button("Rechercher", type="primary", use_container_width=True)

is_bm25 = "BM25" in engine_choice

_, col_filters_main, _ = st.columns([1, 2, 1])

with col_filters_main:
    with st.expander("Filtres avancés", expanded=False):
        
        f_col1, f_col2 = st.columns(2)
        
        with f_col1:
            all_genres = df["Genres"].dropna().astype(str).str.split(",").explode()
            genres_clean = sorted({g.strip() for g in all_genres if g.strip()})
            genres = ["Tous"] + genres_clean
            
            genre = st.selectbox("Genre", genres)
            years = ["Toutes"] + sorted(df["Release_Date"].astype(str).str[:4].dropna().unique(), reverse=True)
            year = st.selectbox("Année", years)
            rating = st.slider("Note minimale", 0.0, 10.0, 6.0, 0.1)
            
        with f_col2:
            duration = st.slider("Durée (min)", 60, 300, (90, 180))
            director = st.text_input("Réalisateur")
            actor = st.text_input("Acteur")

if query and search:
    with st.spinner(f"Recherche avec **{'BM25' if is_bm25 else 'BERT'}**..."):
        if is_bm25:
            results_df = run_search(query, top_n=30)
            if results_df.empty:
                st.error("Aucun résultat BM25")
                st.stop()
            results = results_df.to_dict("records")
        else:
            raw = semantic_search(query=query)
            results = raw if isinstance(raw, list) else []

        filtered = []
        for film in results:
            t = film.get("Title") if isinstance(film, dict) else getattr(film, "Title",str(film))
            
            match = df[df["Title"] == t]
            if match.empty: continue
            row = match.iloc[0]

            if genre != "Tous" and genre not in str(row.get("Genres", "")): continue
            if year != "Toutes" and str(row.get("Release_Date", "")[:4]) != year: continue
            if float(row.get("Vote_Average", 0)) < rating: continue
            if not (duration[0] <= row.get("Runtime", 0) <= duration[1]): continue
            if director and director.lower() not in str(row.get("Director", "")).lower(): continue
            if actor and actor.lower() not in str(row.get("Cast", "")).lower(): continue

            filtered.append((film, row))

    st.success(f"**{len(filtered)} film(s) trouvé(s)** • Moteur : **{'BM25' if is_bm25 else 'BERT'}**")

    cols = st.columns(4)
    
    for i, (film_data, row) in enumerate(filtered):
        col = cols[i % 4]
        
        with col:
            with st.container(border=True):
                
                poster_url = f"https://image.tmdb.org/t/p/w300{row.get('Poster_Path','')}"
                st.image(poster_url, use_container_width=True)
                
                st.markdown(f"#### {row['Title']}")
                st.caption(f"{row['Release_Date'][:4]} • {row.get('Director','Inconnu')}")
                
                if is_bm25:
                    score = film_data.get('score', 0)
                    progress = min(score / 30.0, 1.0)
                    st.markdown(f":blue[**BM25: {score:.2f}**]")
                else:
                    sim = film_data.get("Similarity", 0)
                    progress = sim
                    match_label = f"{int(sim*100)}% Match"
                    color_text = ":green" if sim >= 0.7 else ":orange" if sim >= 0.5 else ":red"
                    st.markdown(f"{color_text}[**{match_label}**]")

                st.markdown(f"⭐ **{row.get('Vote_Average',0):.1f}** • ⏱ {row.get('Runtime',0)} min")
                st.progress(progress)
                
                with st.expander("Détails"):
                    budget = int(row.get('budget', 0)) if str(row.get('budget', 0)).isdigit() else 0
                    revenue = int(row.get('revenue', 0)) if str(row.get('revenue', 0)).isdigit() else 0
                    
                    keywords_display = "Aucun"
                    try:
                        k_str = row.get('Keywords', '[]')
                        if isinstance(k_str, str) and k_str.startswith('['):
                            k_list = json.loads(k_str.replace("'", '"'))
                            keywords_display = ", ".join([k['name'] for k in k_list])
                    except:
                        pass

                    st.markdown(f"""
                    **Synopsis:** {row.get('Overview','Pas de résumé')}
                    
                    **Acteurs:** {str(row.get('Cast',''))}  
                    
                    **Budget:** ${budget:,}  

                    **Recettes:** ${revenue:,}  

                    **Mots-clés:** {keywords_display}  
                    """)

else:
    st.info("Recherchez un film !")
st.markdown("---")
