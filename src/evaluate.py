import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from search_engine import search_movies

# --- Fonction d'évaluation d'une requête ---

def evaluate_query(query, relevant_docs, top_n=10):
    """
    query : texte de la requête utilisateur
    relevant_docs : liste des TITRES considérés comme pertinents (ground truth)
    top_n : nombre de documents retournés
    """
    
    print(f"\n\n=== Évaluation de la requête : '{query}' ===")
    
    # 1. Récupérer les résultats du moteur
    results = search_movies(query, top_n=top_n)
    
    returned_docs = list(results["Title"])
    print("\n Documents retournés :")
    for d in returned_docs:
        print(" -", d)
    
    # 2. Préparer les vecteurs pour les métriques
    # y_true = 1 si le document est pertinent, 0 sinon
    # y_pred = 1 si le moteur l’a retourné, 0 sinon
    all_docs = set(returned_docs) | set(relevant_docs)

    y_true = [1 if doc in relevant_docs else 0 for doc in all_docs]
    y_pred = [1 if doc in returned_docs else 0 for doc in all_docs]

    # 3. Calcul des métriques
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # 4. Affichage
    print("\n Matrice de confusion (TP/FP/FN/TN):")
    print(cm)

    print(f"\n Précision : {precision:.3f}")
    print(f" Rappel    : {recall:.3f}")
    print(f" F1-Score  : {f1:.3f}")

    return {
        "query": query,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "returned_docs": returned_docs,
        "relevant_docs": relevant_docs,
    }

# --- Exemples d'évaluation ---
# Ground truth défini manuellement
relevant_1 = ["Interstellar", "Gravity", "The Martian"]
relevant_2 = ["Love Actually", "Notting Hill", "Crazy Rich Asians"]
relevant_3 = ["Iron Man", "Avengers", "Captain America"]

# Évaluations
eval1 = evaluate_query("space adventure", relevant_1)
eval2 = evaluate_query("romantic comedy", relevant_2)
eval3 = evaluate_query("marvel superhero", relevant_3)

# Résumé des résultats
print("\n\n=== Résumé des évaluations ===")
for eval_res in [eval1, eval2, eval3]:
    print(f"Requête : '{eval_res['query']}' | Précision : {eval_res['precision']:.3f} | Rappel : {eval_res['recall']:.3f} | F1 : {eval_res['f1']:.3f}")

