import os
import json
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

DOCS_PATH = "data/Docs/"
MODEL_NAME = "all-mpnet-base-v2"

print("📥 Chargement du modèle d'embeddings...")
model = SentenceTransformer(MODEL_NAME)

documents = []
texts = []

for file in os.listdir(DOCS_PATH):
    if file.endswith(".json"):
        with open(os.path.join(DOCS_PATH, file), "r", encoding="utf-8") as f:
            doc = json.load(f)
            documents.append(doc)

            text = f"""
            Title: {doc.get('Title','')}
            Overview: {doc.get('Overview','')}
            Keywords: {doc.get('Keywords','')}
            Genres: {doc.get('Genres','')}
            Director: {doc.get('Director','')}
            Cast: {doc.get('Cast','')}
            Year: {str(doc.get('Release_Date',''))[:4]} 
            Tagline: {doc.get('Tagline','')}
            """

            texts.append(text)

print(f"✅ {len(texts)} documents chargés.")

# Création des embeddings
print("⚙️ Création des embeddings...")
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

# Sauvegarder embeddings ET métadonnées
with open("data/embeddings.pkl", "wb") as f:
    pickle.dump({"embeddings": embeddings, "documents": documents, "texts": texts}, f)


print("✅ Embeddings sauvegardés : data/embeddings.pkl")
