import numpy as np
from vector_db import MODEL

def search_query(query, index, films, k=10):
    query_vec = MODEL.encode([query], normalize_embeddings=True).astype('float32')
    scores, indices = index.search(query_vec, k * 3)  # on prend plus pour filtrer doublons
    
    results = []
    seen = set()
    
    for score, idx in zip(scores[0], indices[0]):
        if idx >= len(films):
            continue
        film = films[idx]
        title = film.get("Title", "").lower()
        if title in seen:
            continue
        seen.add(title)
        results.append({
            "film": film,
            "score": round(float(score), 4)
        })
        if len(results) >= k:
            break
    
    return results