import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import pickle

DOCS_PATH = "data/Docs/"

print("📥 Chargement des documents JSON...")

documents = []
titles = []

# Lire tous les documents JSON
for file in os.listdir(DOCS_PATH):
    if file.endswith(".json"):
        with open(os.path.join(DOCS_PATH, file), "r", encoding="utf-8") as f:
            data = json.load(f)

            clean_text = data.get("clean_text", "")
            title = data.get("Title", f"Doc_{len(titles)+1}")

            documents.append(clean_text)
            titles.append(title)

print(f"✅ {len(documents)} documents chargés depuis les fichiers JSON.")

# -------------------- Étape 1 : TF-IDF --------------------

print("📊 Calcul du TF-IDF en cours...")

vectorizer = TfidfVectorizer(max_features=10000)
tfidf_matrix = vectorizer.fit_transform(documents)

terms = vectorizer.get_feature_names_out()

print(f"✅ {len(terms)} termes indexés.")

# -------------------- Étape 2 : Index inversé --------------------

print("🔁 Construction de l’index inversé...")

inverted_index = defaultdict(list)

for term_index, term in enumerate(terms):
    doc_indices = tfidf_matrix[:, term_index].nonzero()[0]

    for doc_id in doc_indices:
        weight = tfidf_matrix[doc_id, term_index]

        inverted_index[term].append({
            "doc_id": int(doc_id),
            "title": titles[doc_id],
            "weight": float(weight)
        })

print("✅ Index inversé construit avec succès.")

# -------------------- Étape 3 : Sauvegarde --------------------

print("💾 Sauvegarde des fichiers...")

# Index inversé
with open("data/inverted_index.json", "w", encoding="utf-8") as f:
    json.dump(inverted_index, f, indent=2, ensure_ascii=False)

# Sauvegarde du vectorizer
with open("data/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Sauvegarde de la matrice TF-IDF
with open("data/tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

print("✅ Tout est sauvegardé avec succès dans le dossier data/")
