import pandas as pd
from search_engine import search_movies

# ===========================================================
# 1) Requêtes de test + jugements de pertinence (ground truth)
# ===========================================================

test_queries = {
    
    "nolan": [
        {"title": "Inception"},
        {"title": "Interstellar"},
        {"title": "The Dark Knight"},
        {"title": "Batman Begins"},
        {"title": "The Dark Knight Rises"}
    ],
    "action": [
        {"title": "Inception"},
        {"title": "Avatar"},
        {"title": "The Dark Knight"},
        {"title": "Batman Begins"},
        {"title": "The Dark Knight Rises"},
        {"title": "Iron Man"},
        {"title": "Deadpool"},
        {"title": "The Avengers"},
        {"title": "Guardians of the Galaxy"},
        {"title": "Avengers: Age of Ultron"},
        {"title": "Captain America: Civil War"},
        {"title": "Mad Max: Fury Road"},
        {"title": "Iron Man 3"},
        {"title": "Pirates of the Caribbean: The Curse of the Black Pearl"}
    ],
    "adventure": [
        {"title": "Inception"},
        {"title": "Interstellar"},
        {"title": "Avatar"},
        {"title": "The Avengers"},
        {"title": "Deadpool"},
        {"title": "Guardians of the Galaxy"},
        {"title": "The Lord of the Rings: The Fellowship of the Ring"},
        {"title": "The Lord of the Rings: The Return of the King"},
        {"title": "Pirates of the Caribbean: The Curse of the Black Pearl"},
        {"title": "Back to the Future"}
    ],
    "science fiction": [
        {"title": "Inception"},
        {"title": "Interstellar"},
        {"title": "Avatar"},
        {"title": "The Avengers"},
        {"title": "Guardians of the Galaxy"},
        {"title": "Iron Man"},
        {"title": "Avengers: Age of Ultron"},
        {"title": "Captain America: Civil War"},
        {"title": "Ant-Man"},
        {"title": "Iron Man 2"}
    ],
    "thriller": [
        {"title": "The Dark Knight"},
        {"title": "Shutter Island"},
        {"title": "Inglourious Basterds"},
        {"title": "Se7en"},
        {"title": "Pulp Fiction"}
    ],
    "comedy": [
        {"title": "Deadpool"},
        {"title": "Forrest Gump"},
        {"title": "The Wolf of Wall Street"},
        {"title": "Back to the Future"},
        {"title": "Inside Out"},
        {"title": "Up"}
    ],
    "romance": [
        {"title": "Titanic"},
        {"title": "Forrest Gump"}
    ],
    "batman": [
        {"title": "The Dark Knight"},
        {"title": "Batman Begins"},
        {"title": "The Dark Knight Rises"}
    ],
    "harry potter": [
        {"title": "Harry Potter and the Philosopher's Stone"},
        {"title": "Harry Potter and the Chamber of Secrets"},
        {"title": "Harry Potter and the Prisoner of Azkaban"},
        {"title": "Harry Potter and the Goblet of Fire"},
        {"title": "Harry Potter and the Order of the Phoenix"},
        {"title": "Harry Potter and the Half-Blood Prince"}
    ],
    "marvel": [
        {"title": "The Avengers"},
        {"title": "Iron Man"},
        {"title": "Deadpool"},
        {"title": "Guardians of the Galaxy"},
        {"title": "Avengers: Age of Ultron"},
        {"title": "Captain America: Civil War"},
        {"title": "Ant-Man"},
        {"title": "Iron Man 2"},
        {"title": "Iron Man 3"}
    ],
    "inception": [
        {"title": "Inception"}
    ]
}


# ===========================================================
# 2) Fonctions métriques
# ===========================================================

def precision(tp, fp):
    if tp + fp == 0: return 0
    return tp / (tp + fp)

def recall(tp, fn):
    if tp + fn == 0: return 0
    return tp / (tp + fn)

def f1_score(p, r):
    if p + r == 0: return 0
    return 2 * (p * r) / (p + r)

# ===========================================================
# 3) Exécution de la campagne d’évaluation
# ===========================================================

def evaluate_system(k=10):
    results_table = []

    for query, relevant_list in test_queries.items():
        print(f"\n🔎 Évaluation de la requête : '{query}'")

        # Exécuter ton moteur de recherche
        results = search_movies(query, top_n=k)

        returned_titles = results['Title'].tolist()
        relevant_set = set([item["title"] for item in relevant_list])

        # Comptage
        tp = len([t for t in returned_titles if t in relevant_set])
        fp = len([t for t in returned_titles if t not in relevant_set])
        fn = len([t for t in relevant_set if t not in returned_titles])

        # Calcul des métriques
        p = precision(tp, fp)
        r = recall(tp, fn)
        f1 = f1_score(p, r)

        print(f"   → Pertinents trouvés (TP) : {tp}")
        print(f"   → Faux positifs (FP) : {fp}")
        print(f"   → Faux négatifs (FN) : {fn}")
        print(f"   → Précision : {p:.3f}")
        print(f"   → Rappel : {r:.3f}")
        print(f"   → F1-score : {f1:.3f}")

        results_table.append([query, p, r, f1])

    # DataFrame final
    df = pd.DataFrame(results_table, columns=["Query", "Precision", "Recall", "F1-score"])
    print("\n======= 📊 RÉSUMÉ GLOBAL =======\n")
    print(df.to_string(index=False))

    df.to_csv("../data/evaluation_results.csv", index=False)
    print("\n💾 Résultats enregistrés dans data/evaluation_results.csv")


# ===========================================================
# 4) Exécution directe
# ===========================================================

if __name__ == "__main__":
    evaluate_system()
