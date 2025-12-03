import os
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import spacy
import string

# -------- CONFIG --------
DOCS_PATH = "data/Docs/"
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")

# -------- Load TF-IDF objects --------
print("📥 Chargement du modèle TF-IDF...")
with open("data/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("data/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

print("✅ Modèle chargé avec succès.")

# -------- Load JSON documents --------
def load_documents():
    docs = []
    for file in os.listdir(DOCS_PATH):
        if file.endswith(".json"):
            with open(os.path.join(DOCS_PATH, file), "r", encoding="utf-8") as f:
                doc = json.load(f)
                docs.append(doc)
    return docs

documents = load_documents()
print(f"✅ {len(documents)} documents JSON chargés.")

# -------- Query preprocessing --------
def preprocess_query(query):
    query = query.lower()
    query = query.translate(str.maketrans("", "", string.punctuation))

    doc = nlp(query)
    tokens = [token.lemma_ for token in doc if not token.is_stop or token.like_num]

    return " ".join(tokens)

# -------- Main search function --------
def search_documents(query, top_n=10, genre_filter=None, year_filter=None):

    clean_query = preprocess_query(query)
    print(f"\n🔍 Requête prétraitée : {clean_query}")

    query_vector = vectorizer.transform([clean_query])

    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    ranked_indices = similarities.argsort()[::-1][:top_n * 2]

    results = []

    for idx in ranked_indices:
        doc = documents[idx]

        title = doc.get("Title", "")
        genres = doc.get("Genres", "")
        year = str(doc.get("Release_Date", ""))[:4]
        director = doc.get("Director", "")
        rating = doc.get("Vote_Average", "")

        # Filters
        if genre_filter and genre_filter.lower() not in str(genres).lower():
            continue
        if year_filter and str(year_filter) not in year:
            continue

        results.append({
            "Title": title,
            "Genres": genres,
            "Year": year,
            "Director": director,
            "Rating": rating,
            "Similarity": round(float(similarities[idx]), 4)
        })

        if len(results) == top_n:
            break

    return results


# -------- Test run --------
if __name__ == "__main__":
    print("\n=== 🎬 TEST DU MOTEUR CINEFINDER ===")

    user_query = input("Entrez une requête (ex: space adventure): ")
    genre = input("Filtrer par genre (optionnel) : ")
    year = input("Filtrer par année (optionnel) : ")

    genre = genre if genre.strip() else None
    year = year if year.strip() else None

    results = search_documents(user_query, top_n=10,
                               genre_filter=genre,
                               year_filter=year)

    print("\n📌 Résultats les plus pertinents :\n")

    if not results:
        print("Aucun résultat trouvé.")
    else:
        for i, res in enumerate(results, 1):
            print(f"{i}. {res['Title']} ({res['Year']})")
            print(f"   Genres : {res['Genres']}")
            print(f"   Directeur : {res['Director']}")
            print(f"   Score : {res['Similarity']}\n")
