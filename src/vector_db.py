import chromadb
from chromadb.utils import embedding_functions

# Import absolu → plus jamais d'erreur
MODEL_NAME = "all-MiniLM-L6-v2"

client = chromadb.PersistentClient(path="chroma_db")

def get_or_create_collection():
    emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
    return client.get_or_create_collection(
        name="movies_collection",
        embedding_function=emb_func
    )

def populate_collection_if_empty(collection, films):
    if collection.count() > 0:
        return
    
    print(f"Ajout de {len(films)} films dans ChromaDB...")
    texts = []
    metadatas = []
    ids = []
    
    for i, film in enumerate(films):
        parts = [
            film.get("Title", ""),
            film.get("Overview", ""),
            film.get("Genres", ""),
            film.get("Director", ""),
            film.get("Cast", ""),
            film.get("Tagline", "")
        ]
        text = " ".join(filter(None, parts))
        texts.append(text)
        
        year = film.get("Release_Date", "")[:4] if film.get("Release_Date") else "?"
        metadatas.append({
            "title": film.get("Title", "Inconnu"),
            "year": year,
            "overview": (film.get("Overview") or "")[:500],
            "genres": film.get("Genres", ""),
            "director": film.get("Director", ""),
            "cast": film.get("Cast", ""),
            "tagline": film.get("Tagline", ""),
            "poster": film.get("Poster_Path", ""),
            "source_file": film.get("_source_file", "")
        })
        ids.append(f"film_{i}")
    
    collection.add(documents=texts, metadatas=metadatas, ids=ids)
    print("Base vectorielle prête !")