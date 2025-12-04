from search_engine import search_documents

ground_truth = {
    "sinking boat": ["Titanic"],
    "wizard school": ["Harry Potter and the Philosopher's Stone"],
    "space travel": ["Interstellar", "Gravity"],
    "dream inside dream": ["Inception"],
    "boxing movie": ["Rocky"]
} # Example ground truth; replace with actual data as needed


def evaluate():
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for query, relevant_movies in ground_truth.items():
        results = search_documents(query)

        predicted = [r["Title"] for r in results]
        relevant = relevant_movies

        TP = len(set(predicted) & set(relevant))
        FP = len(set(predicted) - set(relevant))
        FN = len(set(relevant) - set(predicted))

        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)

        if TP + FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

        print(f"\nQuery: {query}")
        print(f" Precision: {precision:.2f}")
        print(f" Recall: {recall:.2f}")
        print(f" F1-score: {f1:.2f}")

    print("\n=== GLOBAL ===")
    print(f"Average Precision: {sum(all_precisions)/len(all_precisions):.2f}")
    print(f"Average Recall: {sum(all_recalls)/len(all_recalls):.2f}")
    print(f"Average F1-score: {sum(all_f1s)/len(all_f1s):.2f}")


if __name__ == "__main__":
    evaluate()
