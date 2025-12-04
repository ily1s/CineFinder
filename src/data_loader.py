import json
import os
from glob import glob

def load_all_films():
    folder = "movies_raw"
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Dossier '{folder}' introuvable ! Crée-le et mets-y tes JSON.")
    
    films = []
    for filepath in sorted(glob(os.path.join(folder, "*.json"))):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                film = json.load(f)
                film["_source_file"] = os.path.basename(filepath)
                films.append(film)
        except Exception as e:
            print(f"Erreur lecture {filepath} : {e}")
    return films

def create_search_text(film):
    parts = [
        film.get("Title", ""),
        film.get("Overview", ""),
        film.get("Genres", ""),
        film.get("Director", ""),
        film.get("Cast", ""),
        film.get("Tagline", "")
    ]
    return " ".join(filter(None, parts))