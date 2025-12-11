import os
import json
import re

DOCS_PATH = "data/Docs/"

# Load all movies
movies = []
for file in os.listdir(DOCS_PATH):
    if file.endswith(".json"):
        with open(os.path.join(DOCS_PATH, file), "r", encoding="utf-8") as f:
            doc = json.load(f)
            title = doc.get("Title", "").strip()
            text = (
                f"{doc.get('Title','')} "
                f"{doc.get('Overview','')} "
                f"{doc.get('Genres','')} "
                f"{doc.get('Keywords','')}"
                f"{doc.get('Director','')} "
                f"{doc.get('Cast','')} "
                f"{str(doc.get('Release_Date',''))[:4]} "
                f"{doc.get('Tagline','')}"
            ).lower()
            movies.append({"title": title, "text": text})

print(f"Loaded {len(movies)} movies.")

# Utility function
def match(text, keywords):
    return any(kw in text for kw in keywords)

# THEMES + KEYWORDS FOR DETECTION
themes = {
    "movie about a ship sinking in the ocean": [
        "ship", "boat", "sinking", "ocean", "sea", "shipwreck", "storm", "voyage"
    ],
    "space survival mission": [
        "space", "astronaut", "mission", "survival", "planet", "nasa"
    ],
    "superhero saving the world": [
        "superhero", "marvel", "dc", "superhuman", "powers", "hero"
    ],
    "time travel adventure": [
        "time travel", "timeline", "time loop", "past", "future", "temporal"
    ],
    "artificial intelligence and robots": [
        "robot", "android", "ai", "artificial intelligence", "cyborg", "machine"
    ],
    "zombie apocalypse": [
        "zombie", "virus", "infection", "apocalypse", "undead"
    ],
    "heist and robbery": [
        "heist", "robbery", "bank robbery", "thieves", "crime", "criminal gang"
    ],
    "magical fantasy world": [
        "magic", "wizard", "witch", "fantasy", "dragon", "fairy", "mythical"
    ],
    "war and military conflict": [
        "war", "soldier", "battle", "army", "military", "ww1", "ww2", "conflict"
    ],
    "romantic comedy": [
        "romance", "romantic", "comedy", "love story", "rom-com"
    ],
}

# Auto-generate ground truth
ground_truth = {}

for theme, keywords in themes.items():
    matched = [
        m["title"]
        for m in movies
        if match(m["text"], keywords)
    ]
    ground_truth[theme] = matched

# Save to file
with open("data/ground_truth.json", "w", encoding="utf-8") as f:
    json.dump(ground_truth, f, indent=4, ensure_ascii=False)