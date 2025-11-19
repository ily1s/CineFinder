import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np

# Charger le corpus nettoyé
print("Chargement du corpus nettoyé...")
data = pd.read_csv("data/clean_corpus.csv")

# Vérification rapide
print(f"{len(data)} documents chargés.")
print(data.head(2))

# Liste des documents (textes prétraités)
documents = data["clean_text"].fillna("").tolist()

# --- Étape 1 : Calcul du TF-IDF ---
print("Calcul du TF-IDF en cours...")

vectorizer = TfidfVectorizer(
    max_features=10000
)  # on limite à 10000 termes les plus fréquents
tfidf_matrix = vectorizer.fit_transform(documents)

# Récupérer le vocabulaire
terms = vectorizer.get_feature_names_out()

print(f"TF-IDF calculé : {len(terms)} termes indexés.")

# --- Étape 2 : Construction de l’index inversé ---
print("Construction de l’index inversé...")

inverted_index = defaultdict(list)

# Pour chaque terme, lister les documents où il apparaît et son poids TF-IDF
for term_index, term in enumerate(terms):
    # Colonnes non nulles pour ce terme (documents contenant le terme)
    doc_indices = tfidf_matrix[:, term_index].nonzero()[0]
    for doc_id in doc_indices:
        weight = tfidf_matrix[doc_id, term_index]
        inverted_index[term].append(
            {
                "doc_id": int(doc_id),
                "title": data.iloc[doc_id]["Title"],
                "weight": float(weight),
            }
        )

print("Index inversé construit avec succès.")

# --- Étape 3 : Sauvegarde de l’index ---
print("Sauvegarde de l’index dans un fichier JSON...")

with open("data/inverted_index.json", "w", encoding="utf-8") as f:
    json.dump(inverted_index, f, indent=2)

# Sauvegarder aussi la matrice TF-IDF et le vectorizer pour usage futur
import pickle

with open("data/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("data/tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

print("Index inversé, vectorizer et matrice TF-IDF sauvegardés dans /data/")
