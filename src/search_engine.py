import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_PATH = "data/Docs/"
MODEL_NAME = "all-mpnet-base-v2"
SIMILARITY_THRESHOLD = 0.35

print("📥 Chargement du modèle...")
model = SentenceTransformer(MODEL_NAME)

# Charger les embeddings
with open("data/embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    doc_embeddings = data["embeddings"]
    documents = data["documents"]
print(f"✅ {len(doc_embeddings)} embeddings chargés.")


# --- Recherche sémantique ---
def search_documents(query, top_N=10, genre_filter=None, year_filter=None):

    query_embedding = model.encode([query])

    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    results = []

    for i, score in enumerate(similarities):

        if score < SIMILARITY_THRESHOLD:
            continue

        doc = documents[i]

        title = doc.get("Title", "")
        genres = doc.get("Genres", "")
        overview = doc.get("Overview", "")
        year = str(doc.get("Release_Date", ""))[:4]
        director = doc.get("Director", "")
        rating = doc.get("Vote_Average", "")

        # Filtres optionnels
        if genre_filter and genre_filter.lower() not in str(genres).lower():
            continue
        if year_filter and str(year_filter) not in year:
            continue

        results.append(
            {
                "Title": title,
                "Year": year,
                "Genres": genres,
                "Overview": overview,
                "Director": director,
                "Rating": rating,
                "Similarity": round(float(score), 4),
            }
        )

    # Trier par pertinence
    results = sorted(results, key=lambda x: x["Similarity"], reverse=True)[:top_N]

    return results


# -------- Test --------
if __name__ == "__main__":

    print("\n=== 🎬 TEST DU MOTEUR SÉMANTIQUE ===")

    user_query = input("Requête : ")
    # genre = input("Genre (optionnel) : ")
    # year = input("Année (optionnel) : ")

    # genre = genre if genre.strip() else None
    # year = year if year.strip() else None

    results = search_documents(
        user_query, top_N=10, genre_filter=None, year_filter=None
    )

    if not results:
        print("\n❌ Aucun résultat pertinent trouvé.")
    else:
        print(f"\n✅ {len(results)} résultats trouvés :\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['Title']} ({r['Year']})")
            print(f"   Synopsis : {r['Overview']}")
            print(f"   Genres : {r['Genres']}")
            print(f"   Directeur : {r['Director']}")
            print(f"   Score : {r['Similarity']}\n")
