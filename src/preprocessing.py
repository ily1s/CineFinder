import os
import json
import spacy
import string

nlp = spacy.load("en_core_web_sm")

DOCS_PATH = "data/Docs/"

def preprocess_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop or token.like_num]

    return " ".join(tokens)


print("📥 Chargement des documents...")

for file in os.listdir(DOCS_PATH):
    if file.endswith(".json"):
        file_path = os.path.join(DOCS_PATH, file)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        title = data.get("Title", "")
        overview = data.get("Overview", "")
        tagline = data.get("Tagline", "")
        genres = data.get("Genres", "")
        director = data.get("Director", "")
        cast = data.get("Cast", "")
        release_date = str(data.get("Release_Date", ""))[:4]

        full_text = f"{title} {overview} {tagline} {genres} {director} {cast} {release_date}"

        cleaned_text = preprocess_text(full_text)

        # Ajouter le texte nettoyé DIRECTEMENT dans le document
        data["clean_text"] = cleaned_text

        # Sauvegarder le document mis à jour
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

print("✅ Tous les documents ont été nettoyés et mis à jour.")
print("✅ Chaque JSON contient maintenant un champ 'clean_text'")
