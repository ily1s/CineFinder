import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
import spacy
import string

# Télécharger les stopwords (une seule fois)
nltk.download("stopwords")

# Charger le modèle spaCy
nlp = spacy.load("en_core_web_sm")

# Charger le corpus et les objets TF-IDF
print("Chargement des données et du modèle TF-IDF...")
data = pd.read_csv("data/clean_corpus.csv")

with open("data/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("data/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

print("Données et modèle chargés avec succès.")


# --- Prétraitement de la requête ---
def preprocess_query(query):
    query = query.lower()
    query = query.translate(str.maketrans("", "", string.punctuation))
    doc = nlp(query)
    tokens = [token.lemma_ for token in doc if not token.is_stop or token.like_num]
    return " ".join(tokens)


# --- Fonction de recherche principale ---
def search_movies(query, top_n=10, genre_filter=None, year_filter=None):
    # Prétraiter la requête
    clean_query = preprocess_query(query)
    print(f"Requête prétraitée : {clean_query}")

    # Transformer la requête en vecteur TF-IDF
    query_vec = vectorizer.transform([clean_query])

    # Calculer la similarité cosinus entre la requête et tous les documents
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Ordonner par score décroissant
    ranked_indices = similarities.argsort()[::-1]

    # Créer le DataFrame des résultats
    results = data.iloc[ranked_indices].copy()
    results["similarity"] = similarities[ranked_indices]

    # Appliquer des filtres optionnels
    if genre_filter:
        results = results[
            results["Genres"].str.contains(genre_filter, case=False, na=False)
        ]
    if year_filter:
        results = results[
            results["Release_Date"].str.contains(str(year_filter), na=False)
        ]

    # Sélectionner les colonnes à afficher
    results = results[
        ["Title", "Genres", "Release_Date", "Director", "Vote_Average", "similarity"]
    ]

    return results.head(top_n)


# --- Exemple d’utilisation ---
if __name__ == "__main__":
    print("=== TEST DU MOTEUR DE RECHERCHE ===")
    user_query = input("Entrez une requête de recherche (ex: science fiction 2020): ")
    genre = input("Filtrer par genre (laisser vide si aucun): ")
    year = input("Filtrer par année (laisser vide si aucun): ")

    # Si champ vide, on ne passe pas de filtre
    genre = genre if genre.strip() else None
    year = year if year.strip() else None

    results = search_movies(user_query, top_n=10, genre_filter=genre, year_filter=year)
    print("\n Résultats les plus pertinents :\n")
    print(results.to_string(index=False))

