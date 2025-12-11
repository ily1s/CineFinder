import json
import math
from search_engine import search_documents

# Load ground truth
with open("data/ground_truth.json", "r") as f:
    ground_truth = json.load(f)

K = 10  # Your search returns top 10 results


def precision_at_k(predicted, relevant, k):
    predicted_k = predicted[:k]
    hits = sum(1 for p in predicted_k if p in relevant)
    return hits / k


def average_precision(predicted, relevant, k):
    score = 0.0
    hits = 0
    for i, p in enumerate(predicted[:k], start=1):
        if p in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant) if relevant else 0


def dcg(predicted, relevant, k):
    dcg = 0.0
    for i, p in enumerate(predicted[:k], start=1):
        if p in relevant:
            dcg += 1 / math.log2(i + 1)
    return dcg


def ndcg(predicted, relevant, k):
    ideal = sorted(relevant, key=lambda x: 1, reverse=True)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    if idcg == 0:
        return 0
    return dcg(predicted, relevant, k) / idcg


def mrr(predicted, relevant):
    for i, p in enumerate(predicted, start=1):
        if p in relevant:
            return 1 / i
    return 0


def evaluate():
    total_p = 0
    total_map = 0
    total_ndcg = 0
    total_mrr = 0
    n = len(ground_truth)

    for query, relevant in ground_truth.items():
        results = search_documents(query)
        predicted = [r["Title"] for r in results]

        p = precision_at_k(predicted, relevant, K)
        ap = average_precision(predicted, relevant, K)
        nd = ndcg(predicted, relevant, K)
        rr = mrr(predicted, relevant)

        total_p += p
        total_map += ap
        total_ndcg += nd
        total_mrr += rr

        print(f"\nQuery: {query}")
        print(f" P@10:   {p:.3f}")
        print(f" MAP@10: {ap:.3f}")
        print(f" NDCG@10: {nd:.3f}")
        print(f" MRR:    {rr:.3f}")

    print("\n=== GLOBAL ===")
    print(f"Avg P@10:   {total_p/n:.3f}")
    print(f"Avg MAP@10: {total_map/n:.3f}")
    print(f"Avg NDCG@10: {total_ndcg/n:.3f}")
    print(f"Avg MRR:    {total_mrr/n:.3f}")


if __name__ == "__main__":
    evaluate()


