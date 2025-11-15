import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# 1. CHARGEMENT DES DONNÉES
# ============================================
df = pd.read_csv("../data/cleaned_movies.csv")
text_cols = ['Title', 'Overview', 'Tagline', 'Genres', 'Keywords', 'Director', 'Cast']
df = df[text_cols].fillna('')

# ============================================
# 2. TEXTE COMPLET (tous les champs égaux)
# ============================================
df['text'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
df = df.reset_index().rename(columns={'index': 'doc_id'})

# ============================================
# 3. PREPROCESSING: Lemmatisation + Stopwords
# ============================================
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
stopwords = nlp.Defaults.stop_words

def preprocess_text(text):
    """
    Étapes de preprocessing:
    - Minuscules
    - Tokenisation
    - Lemmatisation
    - Suppression stopwords et ponctuation
    """
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and token.lemma_ not in stopwords and len(token) > 2
    ]
    return ' '.join(tokens)

# Appliquer le preprocessing
tqdm.pandas()
df['clean_text'] = df['text'].progress_apply(preprocess_text)

# ============================================
# 4. CONSTRUCTION DE L'INDEX INVERSÉ
# ============================================
inverted_index = defaultdict(set)

for idx, text in enumerate(df['clean_text']):
    for word in set(text.split()):
        inverted_index[word].add(idx)

print(f"Index inversé construit: {len(inverted_index)} termes uniques")

# ============================================
# 5. CALCUL TF-IDF
# ============================================
vectorizer = TfidfVectorizer(max_features=10000)
tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
terms = vectorizer.get_feature_names_out()

print(f"Matrice TF-IDF: {tfidf_matrix.shape}")

# ============================================
# 6. RECHERCHE PAR INTERSECTION (AND)
# ============================================
def search_intersection(query, top_n=10):
    """
    Recherche par INTERSECTION: retourne uniquement les documents 
    contenant TOUS les termes de la requête.
    
    Exemple: "tarantino action" → seulement les films qui ont 
    à la fois "tarantino" ET "action"
    """
    # Preprocessing de la requête
    query_clean = preprocess_text(query)
    query_terms = query_clean.split()
    
    if not query_terms:
        print("Aucun terme valide dans la requête")
        return pd.DataFrame()
    
    # INTERSECTION: documents contenant TOUS les termes
    candidate_docs = None
    for term in query_terms:
        if term in inverted_index:
            term_docs = inverted_index[term]
            if candidate_docs is None:
                candidate_docs = term_docs.copy()
            else:
                candidate_docs = candidate_docs.intersection(term_docs)
        else:
            # Si un terme n'existe pas, aucun résultat
            print(f"Terme '{term}' non trouvé dans l'index")
            return pd.DataFrame()
    
    if not candidate_docs:
        print("Aucun document ne contient TOUS les termes de la requête")
        return pd.DataFrame()
    
    candidate_docs = list(candidate_docs)
    print(f"Intersection trouvée: {len(candidate_docs)} documents contiennent tous les termes")
    
    # Calculer la similarité cosinus sur les documents filtrés
    query_vec = vectorizer.transform([query_clean])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix[candidate_docs]).flatten()
    
    # Trier et prendre les top N
    top_indices_in_candidates = cosine_sim.argsort()[-top_n:][::-1]
    actual_indices = [candidate_docs[i] for i in top_indices_in_candidates]
    
    # Préparer les résultats
    results = df.iloc[actual_indices][['Title', 'Overview', 'Genres', 'Director']].copy()
    results['score'] = cosine_sim[top_indices_in_candidates]
    
    return results

# ============================================
# 7. RECHERCHE PAR UNION (OR) - Alternative
# ============================================
def search_union(query, top_n=10):
    """
    Recherche par UNION: retourne les documents contenant 
    AU MOINS UN des termes de la requête (ton code original).
    """
    query_clean = preprocess_text(query)
    query_terms = query_clean.split()
    
    if not query_terms:
        return pd.DataFrame()
    
    # UNION: documents contenant au moins un terme
    candidate_docs = set()
    for term in query_terms:
        if term in inverted_index:
            candidate_docs.update(inverted_index[term])
    
    if not candidate_docs:
        print("Aucun document trouvé")
        return pd.DataFrame()
    
    candidate_docs = list(candidate_docs)
    print(f"Union trouvée: {len(candidate_docs)} documents contiennent au moins un terme")
    
    query_vec = vectorizer.transform([query_clean])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix[candidate_docs]).flatten()
    
    top_indices_in_candidates = cosine_sim.argsort()[-top_n:][::-1]
    actual_indices = [candidate_docs[i] for i in top_indices_in_candidates]
    
    results = df.iloc[actual_indices][['Title', 'Overview', 'Genres', 'Director']].copy()
    results['score'] = cosine_sim[top_indices_in_candidates]
    
    return results


# ============================================
# 9. TESTS COMPARATIFS
# ============================================
print("\n" + "="*60)
print("TEST: 'tarantino action'")
print("="*60)

print("\n--- INTERSECTION (AND): Tarantino ET Action ---")
results_and = search_intersection("tarantino action", top_n=10)
if not results_and.empty:
    print(results_and[['Title', 'Director', 'Genres', 'score']])

print("\n--- UNION (OR): Tarantino OU Action ---")
results_or = search_union("tarantino action", top_n=10)
if not results_or.empty:
    print(results_or[['Title', 'Director', 'Genres', 'score']])



print("\n" + "="*60)
print("TEST: 'nolan batman dark'")
print("="*60)

print("\n--- INTERSECTION (tous les termes) ---")
results = search_intersection("nolan batman dark", top_n=5)
if not results.empty:
    print(results[['Title', 'Director', 'score']])
